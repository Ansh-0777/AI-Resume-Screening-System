"""
parser.py
Extracts raw text from uploaded resumes / job descriptions (PDF, DOCX, TXT)
and pulls out structured fields (name, email, phone, skills, education,
experience, certifications) from resume text.
"""

import io
import re

import pdfplumber
import docx

from src.utils import extract_email, extract_phone, extract_name, extract_skills


def extract_text_from_pdf(file) -> str:
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text_from_txt(file) -> str:
    raw = file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return raw


def extract_text(file, filename: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    filename = filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    if filename.endswith(".docx"):
        return extract_text_from_docx(file)
    if filename.endswith(".txt"):
        return extract_text_from_txt(file)
    raise ValueError(f"Unsupported file type: {filename}")


SECTION_HEADERS = {
    "education": ["education", "academic background", "qualifications"],
    "experience": ["experience", "work experience", "employment history", "professional experience"],
    "certifications": ["certifications", "certificates", "licenses"],
}


def _extract_section(text: str, keywords: list) -> str:
    """
    Grab the text block that follows a section header (e.g. 'Experience')
    up until the next all-caps / title-like header line.
    """
    lines = text.splitlines()
    lower_lines = [l.strip().lower() for l in lines]

    start_idx = None
    for i, line in enumerate(lower_lines):
        if any(line.startswith(k) or line == k for k in keywords):
            start_idx = i + 1
            break

    if start_idx is None:
        return "Not found"

    collected = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped and (stripped.isupper() or stripped.endswith(":")) and len(stripped.split()) <= 5:
            # looks like the start of a new section header
            if stripped.lower() not in keywords:
                break
        if stripped:
            collected.append(stripped)
        if len(collected) >= 8:
            break

    return "\n".join(collected) if collected else "Not found"


def parse_resume(file, filename: str, skills_df=None) -> dict:
    """
    Full resume parse: raw text plus structured fields.
    Returns a dict ready to feed into scoring/recommendation modules.
    """
    text = extract_text(file, filename)

    return {
        "filename": filename,
        "raw_text": text,
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text, skills_df),
        "education": _extract_section(text, SECTION_HEADERS["education"]),
        "experience": _extract_section(text, SECTION_HEADERS["experience"]),
        "certifications": _extract_section(text, SECTION_HEADERS["certifications"]),
    }


def parse_job_description(file, filename: str, skills_df=None) -> dict:
    """Parse a job description into raw text plus required skills."""
    text = extract_text(file, filename)
    return {
        "filename": filename,
        "raw_text": text,
        "required_skills": extract_skills(text, skills_df),
    }
