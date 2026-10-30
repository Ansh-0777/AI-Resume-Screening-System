"""
similarity.py
TF-IDF vectorization and cosine similarity between a resume and a job
description (or a batch of resumes for ranking).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Returns a 0-100 similarity score between one resume and one JD,
    using TF-IDF + cosine similarity.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(sim) * 100, 2)


def rank_resumes(resume_texts: dict, jd_text: str) -> list:
    """
    resume_texts: {filename: preprocessed_text}
    Returns a list of (filename, score) sorted descending, ranked 1..N.
    Uses a single shared TF-IDF vocabulary (fit across all resumes + JD)
    so scores are comparable across candidates.
    """
    if not resume_texts or not jd_text.strip():
        return []

    filenames = list(resume_texts.keys())
    corpus = [resume_texts[f] for f in filenames] + [jd_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]

    sims = cosine_similarity(resume_vectors, jd_vector)[:, 0]

    ranked = sorted(
        [(filenames[i], round(float(sims[i]) * 100, 2)) for i in range(len(filenames))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked
