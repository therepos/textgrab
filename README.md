## Usage

Extract text from PDFs and images privately. Convert bank statements to categorized CSV.

## Structure

```
TEXTGRAB [GITHUB]
├── .github/
├── .scraps/
├── app/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── dbs_cc.py
│   │   ├── dbs_deposit.py
│   │   ├── generic.py
│   │   └── helpers.py
│   ├── categorize.py
│   ├── extract.py
│   ├── main.py
│   └── rules.json
├── static/
│   └── index.html
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Notes

- Categorise transactions
