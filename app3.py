import pandas as pd
import streamlit as st
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="💼 Job Recommender", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/2d/54/a2/2d54a26774b89feca5464cf43e97933f.jpg");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center top;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255,255,255,0.3);
    z-index: -1;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.9);}
h1, h2, h3, h4, h5, h6, p, label {
    color: #000000 !important;
    background-color: rgba(255, 255, 255, 0.7);
    padding: 4px 8px;
    border-radius: 6px;
}
.stButton>button {
    background-color: #0078D7;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.title("💼 Job Recommendation System")

jobs = pd.read_csv("job_descriptions.csv")[["uniq_id","job_title","job_description","sector","location","organization"]]
resumes = pd.read_csv("Resume.csv")[["ID","Resume_str","Category"]]

combined_text = pd.concat([jobs["job_description"], resumes["Resume_str"]])
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(combined_text)

if st.button("🛠 Generate Similarities"):

    resume_vecs = vectorizer.transform(resumes["Resume_str"])
    job_vecs = vectorizer.transform(jobs["job_description"])
    similarity_matrix = cosine_similarity(resume_vecs, job_vecs)

    joblib.dump(similarity_matrix, "similarity.joblib")
    st.success("✅ Similarity matrix generated and saved as similarity.joblib")


if os.path.exists("similarity.joblib"):
    similarity_matrix = joblib.load("similarity.joblib")
else:
    similarity_matrix = cosine_similarity(vectorizer.transform(resumes["Resume_str"]),
                                          vectorizer.transform(jobs["job_description"]))
    joblib.dump(similarity_matrix, "similarity.joblib")

def recommend_jobs(resume_index, top_n=5):
    scores = list(enumerate(similarity_matrix[resume_index]))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for i, score in ranked:
        results.append({
            "job_title": jobs.iloc[i]["job_title"],
            "organization": jobs.iloc[i]["organization"],
            "location": jobs.iloc[i]["location"],
            "sector": jobs.iloc[i]["sector"],
            "similarity_score": round(score, 3)
        })
    return pd.DataFrame(results)

st.sidebar.header("📌 Options")
choice = st.sidebar.radio("Navigate", ["📘 Learn Skills", "🏢 Company View", "🔗 Related Jobs", "📊 Resume Insights"])

resume_ids = resumes["ID"].tolist()
selected_resume = st.selectbox("📄 Select Resume ID", resume_ids)

if st.button("✨ Recommend Jobs"):
    index = resumes[resumes["ID"] == selected_resume].index[0]
    recommendations = recommend_jobs(index, top_n=5)
    st.subheader(f"🔎 Recommendations for Resume ID {selected_resume}")
    st.dataframe(recommendations)

    if choice == "📘 Learn Skills":
        st.sidebar.image("https://i.pinimg.com/736x/9e/d4/b8/9ed4b8e243d1c0d483e357c66089d6df.jpg", use_container_width=True)
        st.sidebar.info("Explore skill-building resources like Python, SQL, ML basics.")

    elif choice == "🏢 Company View":
        st.sidebar.image("https://i.pinimg.com/736x/b9/f1/76/b9f176bac9429df8bd3c20b93d37195c.jpg", use_container_width=True)
        st.sidebar.dataframe(jobs[["organization","location","sector"]].drop_duplicates().head(10))

    elif choice == "🔗 Related Jobs":
        sector = recommendations["sector"].iloc[0]
        st.sidebar.write(f"Jobs in sector: {sector}")
        st.sidebar.image("https://i.pinimg.com/736x/7c/49/78/7c49786c1aab1724e0162088837498be.jpg", use_container_width=True)
        st.sidebar.dataframe(jobs[jobs["sector"] == sector][["job_title","organization","location"]].head(5))

    elif choice == "📊 Resume Insights":
        st.sidebar.image("https://i.pinimg.com/1200x/17/10/74/171074b9fbb51c97193c75ef71d9fa23.jpg", use_container_width=True)
        category = resumes[resumes["ID"] == selected_resume]["Category"].values[0]
        st.sidebar.success(f"Resume Category: {category}")
        st.sidebar.write("Total resumes loaded:", len(resumes))
