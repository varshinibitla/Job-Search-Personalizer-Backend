# Job Search Personalizer Backend

Backend service that accepts resume text from a frontend app and returns top job recommendations.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip`
- Internet access on first run (to download datasets and NLP/model assets)
- Kaggle access through `kagglehub` (for dataset download)

Required Python packages:

- `pandas`
- `spacy`
- `nltk`
- `scikit-learn`
- `sentence-transformers`
- `kagglehub`

## Setup

1. Create and activate virtual environment:

```bash
python -m venv .venv
```


2. Install dependencies:

```bash
pip install pandas spacy nltk scikit-learn sentence-transformers kagglehub
```

3. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Run Server

From project root:

```bash
python packages/JobSearch.py
```

Optional environment variables:

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8080`)

Example:

```bash
$env:HOST="127.0.0.1"
$env:PORT="8080"
python packages/JobSearch.py
```

## API

### Health / Availability

The server listens continuously once started.

### Endpoint

- Method: `POST`
- Path: `/recommend`
- Content-Type: `application/json`

Request body:

```json
{
  "resume_text": "Python developer with 2 years of ML and SQL experience.",
  "top_n": 5
}
```

Request fields:

- `resume_text` (string, required)
- `top_n` (integer, optional, default `5`)

Successful response:

```json
{
  "results": [
    {
      "title": "Data Scientist",
      "company_name": "Example Inc",
      "location": "New York, NY",
      "score": 0.82
    }
  ]
}
```

Error responses:

- `400`: invalid JSON, missing `resume_text`, or invalid `top_n`
- `404`: route not found

## Frontend Example (fetch)

```javascript
const response = await fetch("http://localhost:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    resume_text: resumeTextFromForm,
    top_n: 5
  })
});

const data = await response.json();
console.log(data.results);
```

## Notes

- First startup can be slow because models and datasets are initialized.
- The API includes permissive CORS headers (`*`) for local frontend integration.
