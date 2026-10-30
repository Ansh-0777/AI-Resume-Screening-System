"""
AI-Powered Resume Screening System
Streamlit front-end. Run with: streamlit run app.py
"""

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.parser import parse_resume, parse_job_description
from src.preprocessing import preprocess
from src.similarity import compute_similarity, rank_resumes
from src.scoring import compute_match_score, compute_ats_score
from src.recommendation import generate_skill_recommendations, generate_resume_tips, generate_ai_summary
from src.utils import load_skills_database, missing_skills, matched_skills, score_label

st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "resumes" not in st.session_state:
    st.session_state.resumes = {}          # {filename: parsed_resume_dict}
if "jd" not in st.session_state:
    st.session_state.jd = None             # parsed JD dict
if "results" not in st.session_state:
    st.session_state.results = {}          # {filename: {score, ats, missing, matched}}

skills_df = load_skills_database()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📄 Resume Screening")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Upload Resume(s)", "Upload Job Description", "Analysis", "Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Resumes loaded: **{len(st.session_state.resumes)}**")
st.sidebar.caption(f"Job description loaded: **{'Yes' if st.session_state.jd else 'No'}**")

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("AI-Powered Resume Screening System")
    st.markdown(
        """
        This tool automatically analyzes resumes against a job description,
        calculates a compatibility score, flags missing skills, and
        recommends concrete improvements — for both recruiters screening
        candidates and applicants tuning their resume.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Resumes uploaded", len(st.session_state.resumes))
    col2.metric("Job description", "Loaded" if st.session_state.jd else "Not loaded")
    col3.metric("Analyses run", len(st.session_state.results))

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Upload Resume(s)** — PDF or DOCX, one or many (for ranking).
        2. **Upload Job Description** — PDF, DOCX, or TXT.
        3. **Analysis** — match score, missing skills, AI-style recommendations.
        4. **Dashboard** — charts, ranking, and skill-frequency breakdown.
        """
    )

# ---------------------------------------------------------------------------
# Upload Resume(s)
# ---------------------------------------------------------------------------
elif page == "Upload Resume(s)":
    st.title("Upload Resume(s)")
    st.caption("Upload one resume for a single analysis, or several to rank candidates.")

    uploaded = st.file_uploader(
        "Choose resume file(s) (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.resumes:
                try:
                    parsed = parse_resume(io.BytesIO(f.getvalue()), f.name, skills_df)
                    st.session_state.resumes[f.name] = parsed
                    st.success(f"Parsed: {f.name}")
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

    if st.session_state.resumes:
        st.markdown("### Parsed Resumes")
        for name, r in st.session_state.resumes.items():
            with st.expander(name):
                c1, c2 = st.columns(2)
                c1.write(f"**Name:** {r['name']}")
                c1.write(f"**Email:** {r['email']}")
                c1.write(f"**Phone:** {r['phone']}")
                c2.write(f"**Skills found:** {', '.join(r['skills']) if r['skills'] else 'None detected'}")
                st.write(f"**Experience (excerpt):** {r['experience'][:300]}")
                st.write(f"**Education (excerpt):** {r['education'][:300]}")

        if st.button("🗑️ Clear all resumes"):
            st.session_state.resumes = {}
            st.session_state.results = {}
            st.rerun()

# ---------------------------------------------------------------------------
# Upload Job Description
# ---------------------------------------------------------------------------
elif page == "Upload Job Description":
    st.title("Upload Job Description")

    uploaded = st.file_uploader("Choose a JD file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    pasted = st.text_area("...or paste the job description text directly", height=200)

    if uploaded is not None:
        try:
            parsed = parse_job_description(io.BytesIO(uploaded.getvalue()), uploaded.name, skills_df)
            st.session_state.jd = parsed
            st.success(f"Parsed job description: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to parse JD: {e}")
    elif pasted.strip():
        from src.utils import extract_skills
        st.session_state.jd = {
            "filename": "pasted_job_description.txt",
            "raw_text": pasted,
            "required_skills": extract_skills(pasted, skills_df),
        }
        st.success("Job description captured from pasted text.")

    if st.session_state.jd:
        st.markdown("### Required Skills Detected")
        req = st.session_state.jd["required_skills"]
        st.write(", ".join(req) if req else "No known skills detected — try pasting a more detailed JD.")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
elif page == "Analysis":
    st.title("Analysis")

    if not st.session_state.resumes:
        st.warning("Upload at least one resume first.")
    elif not st.session_state.jd:
        st.warning("Upload a job description first.")
    else:
        jd_clean = preprocess(st.session_state.jd["raw_text"])
        jd_skills = st.session_state.jd["required_skills"]

        if st.button("🔍 Run Analysis", type="primary"):
            results = {}
            for name, resume in st.session_state.resumes.items():
                resume_clean = preprocess(resume["raw_text"])
                similarity = compute_similarity(resume_clean, jd_clean)
                final_score = compute_match_score(similarity, resume["skills"], jd_skills)
                ats = compute_ats_score(resume)
                results[name] = {
                    "similarity": similarity,
                    "final_score": final_score,
                    "ats_score": ats,
                    "missing": missing_skills(resume["skills"], jd_skills),
                    "matched": matched_skills(resume["skills"], jd_skills),
                }
            st.session_state.results = results
            st.success("Analysis complete — see results below.")

        if st.session_state.results:
            for name, res in st.session_state.results.items():
                resume = st.session_state.resumes[name]
                st.markdown(f"## {name}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Resume Match Score", f"{res['final_score']}%", score_label(res["final_score"]))
                c2.metric("TF-IDF Similarity", f"{res['similarity']}%")
                c3.metric("ATS Compatibility", f"{res['ats_score']} / 100")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Matched Skills**")
                    st.write(", ".join(res["matched"]) if res["matched"] else "None")
                with col_b:
                    st.markdown("**❌ Missing Skills**")
                    st.write(", ".join(res["missing"]) if res["missing"] else "None — full skill coverage!")

                st.markdown("**💡 Learning Recommendations**")
                recs = generate_skill_recommendations(resume["skills"], jd_skills)
                if recs:
                    for r in recs:
                        st.markdown(f"- **{r['missing_skill']}**: {r['recommendation']}")
                else:
                    st.write("No skill gaps detected relative to the job description.")

                st.markdown("**📝 Resume Improvement Tips**")
                for t in generate_resume_tips(resume, jd_skills):
                    st.markdown(f"- {t}")

                st.markdown("**🧾 AI-Style Professional Summary**")
                st.info(generate_ai_summary(resume))

                st.markdown("---")

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
elif page == "Dashboard":
    st.title("Dashboard")

    if not st.session_state.results:
        st.warning("Run an analysis first (see the Analysis page).")
    else:
        # Ranking table
        ranking = sorted(
            [(name, res["final_score"]) for name, res in st.session_state.results.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        rank_df = pd.DataFrame(ranking, columns=["Resume", "Match Score"])
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

        st.markdown("### 🏆 Resume Ranking")
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Match Score by Resume")
            fig_bar = px.bar(rank_df, x="Resume", y="Match Score", color="Match Score",
                              color_continuous_scale="Blues", range_y=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            if len(rank_df) == 1:
                st.markdown("### Match Score Breakdown")
                score = rank_df["Match Score"].iloc[0]
                fig_pie = go.Figure(go.Pie(
                    values=[score, 100 - score],
                    labels=["Match", "Gap"],
                    hole=0.55,
                    marker_colors=["#2563eb", "#e5e7eb"],
                ))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.markdown("### Score Distribution")
                fig_hist = px.histogram(rank_df, x="Match Score", nbins=10)
                st.plotly_chart(fig_hist, use_container_width=True)

        # Skill frequency across all resumes
        st.markdown("### Top Skills Across Uploaded Resumes")
        all_skills = []
        for r in st.session_state.resumes.values():
            all_skills.extend(r["skills"])
        if all_skills:
            freq = pd.Series(all_skills).value_counts().reset_index()
            freq.columns = ["Skill", "Count"]
            fig_skills = px.bar(freq.head(15), x="Count", y="Skill", orientation="h")
            fig_skills.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.write("No skills detected yet.")

        # Missing skills across all resumes
        st.markdown("### Most Common Missing Skills")
        all_missing = []
        for res in st.session_state.results.values():
            all_missing.extend(res["missing"])
        if all_missing:
            miss_freq = pd.Series(all_missing).value_counts().reset_index()
            miss_freq.columns = ["Skill", "Count"]
            fig_missing = px.bar(miss_freq, x="Skill", y="Count", color="Count",
                                  color_continuous_scale="Reds")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.write("No missing skills across the analyzed resumes — nice.")
