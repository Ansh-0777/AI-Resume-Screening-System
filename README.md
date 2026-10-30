# AI Resume Screening System

An AI-powered resume screening application that analyzes resumes, compares
them against a job description, calculates a compatibility score, flags
missing skills, and recommends improvements — built for recruiters
shortlisting candidates and for applicants tuning their resumes.

## Features

- **Resume & JD upload** — PDF, DOCX, and TXT support
- **Resume parsing** — name, email, phone, skills, education, experience, certifications
- **Text cleaning pipeline** — lowercasing, punctuation removal, stopword removal, lemmatization (NLTK)
- **Skill extraction** — matched against a configurable skills database (`data/skills_database.csv`)
- **Resume matching** — TF-IDF vectorization + cosine similarity
- **Missing skill detection** — required vs. present skills
- **AI-style recommendations** — learning suggestions per missing skill, plus general resume tips
- **Resume ranking** — upload multiple resumes, rank them against one JD
- **ATS Compatibility Score** — heuristic 0–100 score for machine-parseability
- **AI Resume Summary** — auto-generated professional summary
- **Interactive dashboard** — ranking table, bar/pie/histogram charts, skill-frequency breakdown

## Tech Stack

| Layer | Tools |
|---|---|
| Backend / Frontend | Python, Streamlit |
| NLP | NLTK, TF-IDF, Cosine Similarity |
| ML | Scikit-learn |
| PDF Processing | pdfplumber |
| DOCX Processing | python-docx |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |

## Project Structure

```
AI-Resume-Screening-System/
├── app.py                     # Streamlit app (entry point)
├── requirements.txt
├── README.md
├── data/
│   ├── resumes/                # (optional) sample resumes
│   ├── job_descriptions/       # (optional) sample JDs
│   └── skills_database.csv     # skill list used for extraction
├── src/
│   ├── parser.py               # PDF/DOCX/TXT text + field extraction
│   ├── preprocessing.py        # cleaning, tokenizing, lemmatizing
│   ├── similarity.py           # TF-IDF + cosine similarity, ranking
│   ├── recommendation.py       # missing-skill tips, resume tips, AI summary
│   ├── scoring.py               # blended match score, ATS score
│   └── utils.py                 # skill DB loading, regex extraction helpers
├── models/                     # (optional) saved models via joblib
├── reports/                    # (optional) exported analysis reports
├── notebooks/                  # exploratory notebooks
└── sample_files/                # sample resumes/JDs for demoing
```

## Installation

```bash
git clone <your-repo-url>
cd AI-Resume-Screening-System
pip install -r requirements.txt
streamlit run app.py
```

On first run, NLTK will automatically download the small corpora it needs
(`punkt`, `stopwords`, `wordnet`). If you're offline, run:

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

## How It Works (Pipeline)

```
Resume/JD file
   → Extract text (pdfplumber / python-docx)
   → Clean text (lowercase, strip punctuation)
   → Tokenize
   → Remove stopwords
   → Lemmatize
   → Extract skills (regex match against skills_database.csv)
   → TF-IDF vectorize (resume + JD)
   → Cosine similarity
   → Blend similarity + skill overlap → Final Match Score
   → Missing-skill detection
   → Recommendations
   → Dashboard / Report
```

## Extending This Project

- Swap the templated `generate_ai_summary()` in `src/recommendation.py` for
  a real LLM call (e.g. the Anthropic API) for a fully generative summary.
- Add OCR (e.g. `pytesseract`) to `parser.py` for scanned/image-based resumes.
- Add a classifier (Logistic Regression / Naive Bayes / KNN) trained on a
  labeled resume dataset for an "Accuracy / Precision / Recall / F1" evaluation
  section, as scikit-learn is already a dependency.
- Swap TF-IDF for `sentence-transformers` embeddings for semantic (not just
  lexical) matching.
- Add authentication and a database (e.g. SQLite) to persist screening runs
  for recruiters across sessions.

## Future Scope

- OCR support for scanned resumes
- LinkedIn profile analysis
- Voice resume input
- Multi-language resume parsing
- Cloud deployment (Streamlit Community Cloud / AWS / Azure)
