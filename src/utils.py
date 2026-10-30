"""
utils.py
Shared helper functions: skills database loading, skill extraction,
and small formatting helpers used across the app.
"""

import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DB_PATH = os.path.join(BASE_DIR, "data", "skills_database.csv")


def load_skills_database(path: str = SKILLS_DB_PATH) -> pd.DataFrame:
    """Load the skills database CSV. Falls back to an empty frame if missing."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["skill", "category"])
    return pd.read_csv(path)


def extract_skills(text: str, skills_df: pd.DataFrame = None) -> list:
    """
    Match known skills (from the skills database) inside a block of text.
    Case-insensitive, whole-word/phrase matching so 'R' doesn't match inside
    'Recruiter', etc.
    """
    if skills_df is None:
        skills_df = load_skills_database()

    if not text:
        return []

    text_lower = text.lower()
    found = []

    for skill in skills_df["skill"].dropna().tolist():
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)

    return sorted(set(found))


def missing_skills(resume_skills: list, jd_skills: list) -> list:
    """Skills required by the JD but absent from the resume."""
    resume_set = {s.lower() for s in resume_skills}
    return [s for s in jd_skills if s.lower() not in resume_set]


def matched_skills(resume_skills: list, jd_skills: list) -> list:
    """Skills present in both the resume and the JD."""
    resume_set = {s.lower() for s in resume_skills}
    return [s for s in jd_skills if s.lower() in resume_set]


def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text or "")
    return match.group(0) if match else "Not found"


def extract_phone(text: str) -> str:
    match = re.search(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{2,5}",
        text or "",
    )
    candidate = match.group(0).strip() if match else ""
    digit_count = sum(ch.isdigit() for ch in candidate)
    return candidate if digit_count >= 7 else "Not found"


def extract_name(text: str) -> str:
    """
    Very lightweight heuristic: assume the candidate's name is on one of the
    first few non-empty lines and doesn't contain digits/@ symbols.
    Good enough for a demo parser; a production system would use NER.
    """
    if not text:
        return "Not found"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4:
            return line
    return "Not found"


def score_label(score: float) -> str:
    """Convert a 0-100 score into a qualitative label."""
    if score >= 80:
        return "Excellent Match"
    if score >= 60:
        return "Good Match"
    if score >= 40:
        return "Moderate Match"
    return "Weak Match"
