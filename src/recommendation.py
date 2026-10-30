"""
recommendation.py
Generates human-readable improvement suggestions: learning recommendations
for missing skills, plus general resume-quality tips based on the parsed
resume and its match against the job description.
"""

from src.utils import missing_skills, matched_skills

# Short, generic learning nudges per skill. Falls back to a generic template
# for any skill not explicitly listed here.
LEARNING_TIPS = {
    "TensorFlow": "Learn TensorFlow fundamentals and build one image classification project.",
    "PyTorch": "Complete a PyTorch beginner course and implement a small neural network from scratch.",
    "Deep Learning": "Study core deep learning concepts (CNNs, RNNs) and add a project demonstrating them.",
    "Git": "Practice Git basics (commit, branch, merge) and host a project on GitHub.",
    "GitHub": "Publish at least one project on GitHub with a clear README.",
    "Docker": "Containerize one of your existing projects with Docker and document the steps.",
    "AWS": "Complete an AWS Cloud Practitioner tutorial and deploy a small project on AWS.",
    "SQL": "Practice SQL queries on a sample database and add a data-driven project to your portfolio.",
    "Power BI": "Build a small interactive dashboard in Power BI using public data.",
    "Machine Learning": "Complete an applied ML course and add an end-to-end ML project to your portfolio.",
}


def generate_skill_recommendations(resume_skills: list, jd_skills: list) -> list:
    """
    For each skill required by the JD but missing from the resume, produce
    a 'Missing Skill / Recommendation' pair.
    """
    missing = missing_skills(resume_skills, jd_skills)
    recs = []
    for skill in missing:
        tip = LEARNING_TIPS.get(
            skill,
            f"Learn the fundamentals of {skill} and showcase it in a small project.",
        )
        recs.append({"missing_skill": skill, "recommendation": tip})
    return recs


def generate_resume_tips(resume: dict, jd_skills: list) -> list:
    """General resume-quality suggestions based on parsed sections."""
    tips = []

    if resume.get("email") in (None, "Not found", ""):
        tips.append("Add a professional email address near the top of your resume.")
    if resume.get("phone") in (None, "Not found", ""):
        tips.append("Include a reachable phone number in your contact details.")
    if resume.get("experience") in (None, "Not found", ""):
        tips.append("Add a clearly labeled 'Experience' section describing your roles and impact.")
    if resume.get("education") in (None, "Not found", ""):
        tips.append("Add a clearly labeled 'Education' section with degree, institution, and year.")
    if resume.get("certifications") in (None, "Not found", ""):
        tips.append("Consider adding relevant certifications, if you have any, in their own section.")

    matched = matched_skills(resume.get("skills", []), jd_skills)
    if matched:
        top = ", ".join(matched[:3])
        tips.append(f"Highlight your experience with {top} more prominently — it's directly relevant to this role.")

    if not tips:
        tips.append("Your resume covers the core sections well — focus on quantifying achievements with numbers.")

    return tips


def generate_ai_summary(resume: dict) -> str:
    """
    Short, templated professional summary based on extracted skills.
    (A stand-in for an LLM-generated summary — swap in an API call to
    Claude/GPT here if you want a fully generative version.)
    """
    name = resume.get("name", "The candidate")
    skills = resume.get("skills", [])
    top_skills = ", ".join(skills[:5]) if skills else "a range of technical skills"

    return (
        f"{name} is a professional with hands-on experience in {top_skills}. "
        f"Their background reflects a practical, project-driven approach, "
        f"and their resume highlights relevant technical and domain skills "
        f"aligned with modern industry expectations."
    )
