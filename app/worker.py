"""Background worker and input analysis for the job queue.

- worker_loop(): single coroutine that claims queued jobs and runs the
  scheme transform, one at a time. GPU-safe by construction.
- analyse_pdf(): inspects raw PDF bytes and returns (page_count, ocr_expected)
  without actually running Docling. Used to price ETA at submission time.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Dict, Optional, Tuple

from .jobs import JobStore
from .schemes import get_scheme

logger = logging.getLogger(__name__)


def analyse_pdf(content: bytes) -> Tuple[Optional[int], bool]:
    """Return (page_count, ocr_expected_bool) for a PDF.

    'ocr_expected' is True when >50% of pages have no extractable text —
    heuristic signal that Docling will invoke its OCR engine.

    Returns (None, False) if the bytes aren't a recognisable PDF.
    """
    try:
        import pdfplumber
    except Exception as e:
        logger.warning(f"pdfplumber unavailable: {e}")
        return None, False

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            total = len(pdf.pages)
            if total == 0:
                return 0, False
            # Cheap signal: count pages that have any chars at all.
            sample_size = min(total, 10)
            stride = max(1, total // sample_size)
            sparse_pages = 0
            sampled = 0
            for i in range(0, total, stride):
                p = pdf.pages[i]
                if len(p.chars) < 20:
                    sparse_pages += 1
                sampled += 1
                if sampled >= sample_size:
                    break
            ocr_likely = sampled > 0 and (sparse_pages / sampled) > 0.5
            return total, ocr_likely
    except Exception as e:
        logger.warning(f"analyse_pdf failed: {e}")
        return None, False


def analyse_inputs(scheme: str, inputs: Dict[str, bytes]) -> Tuple[int, bool]:
    """Aggregate analyse_pdf over all inputs for a job.
    Returns (total_pages, any_ocr_expected)."""
    total_pages = 0
    any_ocr = False
    for fname, content in inputs.items():
        if fname.lower().endswith(".pdf"):
            pg, ocr = analyse_pdf(content)
            if pg:
                total_pages += pg
            if ocr:
                any_ocr = True
    return total_pages, any_ocr


async def worker_loop(store: JobStore, stop_event: asyncio.Event):
    """Single-consumer queue worker. Picks one queued job at a time, runs
    it to completion on a thread pool (scheme transforms are blocking),
    writes the result back."""
    logger.info("job worker started")
    try:
        while not stop_event.is_set():
            job = None
            try:
                # Try to claim a job from the queue
                job = await asyncio.to_thread(store.claim_next_queued)
            except Exception:
                logger.exception("claim_next_queued failed; retrying")

            if job is None:
                # Idle poll — 1s is plenty for a UI-driven queue
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            logger.info(f"job {job.id} claimed (scheme={job.scheme}, pages={job.page_count})")
            try:
                inputs = await asyncio.to_thread(store.load_inputs, job.id)
                if inputs is None:
                    await asyncio.to_thread(
                        store.mark_failed, job.id,
                        "input blob missing (job record corrupted)",
                    )
                    continue

                scheme_mod = get_scheme(job.scheme)
                if scheme_mod is None:
                    await asyncio.to_thread(
                        store.mark_failed, job.id,
                        f"scheme '{job.scheme}' not found (no longer registered?)",
                    )
                    continue

                # Run the transform on a worker thread. Docling releases
                # the GIL during native inference, but the transform is
                # definitely blocking for our purposes.
                result = await asyncio.to_thread(
                    scheme_mod.transform, inputs, job.output_mode,
                )

                # Extract which engine actually ran (Docling vs heuristic)
                engine_used = _detect_engine_from_result(result)
                await asyncio.to_thread(store.mark_done, job.id, result, engine_used)
                logger.info(f"job {job.id} done (engine={engine_used})")
            except Exception as e:
                logger.exception(f"job {job.id} failed")
                try:
                    await asyncio.to_thread(
                        store.mark_failed, job.id, f"{type(e).__name__}: {e}",
                    )
                except Exception:
                    logger.exception("mark_failed also failed")
    finally:
        logger.info("job worker stopping")


def _detect_engine_from_result(result: dict) -> Optional[str]:
    """Inspect a pdf2md-style result envelope to determine which engine
    produced each page. Returns 'docling', 'heuristic', 'mixed', or None.
    Other schemes (mhtml2md, etc.) return None."""
    if not isinstance(result, dict):
        return None
    conf = result.get("confidence") or {}
    pages = conf.get("pages") or []
    if not pages:
        return None
    engines = set()
    for p in pages:
        e = p.get("engine")
        if e:
            engines.add(e)
    if len(engines) == 1:
        return next(iter(engines))
    if engines:
        return "mixed"
    return None


async def cleanup_loop(store: JobStore, stop_event: asyncio.Event,
                       interval: float = 3600.0):
    """Periodic retention cleanup."""
    logger.info(f"cleanup loop started (interval={interval}s)")
    try:
        # Run once on startup
        try:
            await asyncio.to_thread(store.cleanup_expired)
        except Exception:
            logger.exception("startup cleanup failed")

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                break
            try:
                await asyncio.to_thread(store.cleanup_expired)
            except Exception:
                logger.exception("cleanup_expired failed")
    finally:
        logger.info("cleanup loop stopping")
