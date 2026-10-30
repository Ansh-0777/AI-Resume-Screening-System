"""
scoring.py
Combines similarity score + skill overlap into a final resume match score,
plus a simple heuristic ATS (Applicant Tracking System) compatibility score.
"""


def compute_match_score(similarity_score: float, resume_skills: list, jd_skills: list,
                         similarity_weight: float = 0.6, skill_weight: float = 0.4) -> float:
    """
    Blend TF-IDF/cosine similarity with skill overlap into one 0-100 score.
    Skill overlap = (# matched skills) / (# required skills), guarding for
    JDs where no known skills were detected.
    """
    if jd_skills:
        resume_set = {s.lower() for s in resume_skills}
        matched = [s for s in jd_skills if s.lower() in resume_set]
        skill_overlap = (len(matched) / len(jd_skills)) * 100
    else:
        skill_overlap = similarity_score  # fall back if no skills detected in JD

    final_score = (similarity_score * similarity_weight) + (skill_overlap * skill_weight)
    return round(final_score, 2)


def compute_ats_score(resume: dict) -> int:
    """
    Lightweight heuristic ATS-compatibility checker (0-100).
    Rewards presence of standard, machine-parseable fields and penalizes
    missing contact info or empty sections -- a stand-in for a full ATS
    parser, good enough to flag obviously incomplete resumes.
    """
    score = 0
    checks = 0

    def check(condition, points):
        nonlocal score, checks
        checks += 1
        if condition:
            score += points

    check(resume.get("email") not in (None, "Not found", ""), 20)
    check(resume.get("phone") not in (None, "Not found", ""), 15)
    check(resume.get("name") not in (None, "Not found", ""), 15)
    check(resume.get("education") not in (None, "Not found", ""), 15)
    check(resume.get("experience") not in (None, "Not found", ""), 20)
    check(len(resume.get("skills", [])) >= 3, 15)

    return score
