import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Load Data
# ------------------------
news_articles = pickle.load(open('news_articles.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
user_interest = pickle.load(open('news_articles.pkl', 'rb'))
user_ratings = pickle.load(open('user_rated_articles.pkl', 'rb'))

# ------------------------
# Identify Columns
# ------------------------
def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

TITLE_COL = find_col(news_articles, ['title', 'headline'])
DESC_COL = find_col(news_articles, ['description', 'content'])
URL_COL = find_col(news_articles, ['url', 'link'])

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="Smart News", layout="centered")

# ------------------------
# CSS Styling
# ------------------------
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #a4508b, #5f0a87);
  color: white;
  font-family: "Inter", sans-serif;
}
.header-title {
  text-align:center;
  font-size:58px;
  font-weight:900;
  margin-top:10px;
}
.header-sub {
  text-align:center;
  color:#e0d2ff;
  font-size:22px;
  margin-bottom:20px;
}

/* Toggle box styling */
.toggle-box {
  display:flex;
  justify-content:center;
  align-items:center;
  gap:15px;
  background:rgba(255,255,255,0.15);
  border-radius:40px;
  padding:10px 18px;
  width:fit-content;
  margin:0 auto 35px auto;
  box-shadow:0 6px 20px rgba(0,0,0,0.3);
}
.toggle-btn {
  border:none;
  border-radius:25px;
  padding:12px 30px;
  font-weight:600;
  cursor:pointer;
  background:transparent;
  color:white;
}
.toggle-btn:hover {
  background:rgba(255,255,255,0.25);
}
.active-btn {
  background:white;
  color:#5f0a87;
  box-shadow:0 4px 15px rgba(0,0,0,0.25);
}

/* Search box and hashtags */
.search-box {
  text-align:center;
  background:rgba(255,255,255,0.12);
  padding:30px;
  border-radius:20px;
  width:80%;
  margin:0 auto 30px auto;
  box-shadow:0 6px 20px rgba(0,0,0,0.3);
}
.hashtags {
  text-align:center;
  margin-top:12px;
}
.hashtag-btn {
  background:rgba(255,255,255,0.15);
  color:white;
  border:none;
  border-radius:22px;
  padding:8px 18px;
  margin:8px;
  cursor:pointer;
  font-weight:500;
  transition:0.2s;
}
.hashtag-btn:hover {
  background:rgba(255,255,255,0.3);
  transform:scale(1.05);
}

/* News cards */
.card {
  background:rgba(255,255,255,0.08);
  border-radius:14px;
  padding:18px;
  margin-bottom:15px;
  box-shadow:0 6px 18px rgba(0,0,0,0.25);
}
.card-title { font-weight:700; font-size:18px; color:white; margin-bottom:6px; }
.card-desc { color:#e0d2ff; margin-bottom:8px; }

/* Back button */
div[data-testid="stButton"] > button.back-home {
  background: rgba(255,255,255,0.2);
  color:white;
  border:none;
  border-radius:20px;
  padding:8px 16px;
  position:absolute;
  top:20px;
  right:40px;
}
div[data-testid="stButton"] > button.back-home:hover {
  background: rgba(255,255,255,0.35);
}

/* Gradient buttons */
div[data-testid="stButton"] > button:not(.back-home) {
  background: linear-gradient(90deg,#b86bd4,#7a1ea1);
  color:white;
  border-radius:25px;
  padding:10px 22px;
  font-weight:600;
  border:none;
}
div[data-testid="stButton"] > button:not(.back-home):hover {
  transform:scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Session States
# ------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "home"

if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False

if "selected_hashtag" not in st.session_state:
    st.session_state.selected_hashtag = ""

# ------------------------
# Helper Function
# ------------------------
def back_to_home():
    st.session_state.mode = "home"
    st.session_state.search_clicked = False
    st.session_state.selected_hashtag = ""

# ------------------------
# LANDING PAGE
# ------------------------
if st.session_state.mode == "home":
    # Main Title
    st.markdown("<div class='header-title'>Smart News</div>", unsafe_allow_html=True)

    # Subtitle
    st.markdown("""
    <div class='header-sub' style='margin-top:-10px; font-size:22px;'>
        Discover personalized news recommendations powered by advanced AI technology
    </div>
    """, unsafe_allow_html=True)

    # Toggle Box with two buttons inside
    # st.markdown("<div class='toggle-box'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" News Title Based", key="btn_title"):
            st.session_state.mode = "title"
    with col2:
        if st.button(" Personalized", key="btn_personalized"):
            st.session_state.mode = "personalized"
    st.markdown("</div>", unsafe_allow_html=True)

    # Section heading - Discover News That Matters
    st.markdown("""
    <div style='text-align:center; font-size:36px; font-weight:800; margin-top:30px;'>
        Discover News That Matters
    </div>
    """, unsafe_allow_html=True)

    # Subheading below it
    st.markdown("""
    <div style='text-align:center; color:#e0d2ff; font-size:20px; margin-bottom:25px;'>
        AI-powered recommendations from thousands of sources
    </div>
    """, unsafe_allow_html=True)

    # Search + Hashtags box
    # st.markdown("<div class='search-box'>", unsafe_allow_html=True)
    query = st.text_input(
        "Search News:",
        value=st.session_state.selected_hashtag,
        placeholder="Type a keyword or click a hashtag below..."
    )

    # Hashtags section
    hashtags = ["sports", "health", "finance", "entertainment", "technology", "politics", "science"]
    cols = st.columns(4)
    for i, tag in enumerate(hashtags):
        if cols[i % 4].button(f"#{tag}", key=f"tag_{tag}"):
            st.session_state.selected_hashtag = tag
            st.session_state.search_clicked = True
            query = tag

    # Manual search button
    if st.button("🔍 Search"):
        st.session_state.search_clicked = True
        st.session_state.selected_hashtag = query.strip()

    st.markdown("</div>", unsafe_allow_html=True)

    # Show search results
    if st.session_state.search_clicked and query:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(news_articles[TITLE_COL].astype(str))
        query_vec = tfidf.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-10:][::-1]
        results = news_articles.iloc[top_indices]

        if results.empty:
            st.info("No matches found. Showing trending articles.")
            results = news_articles.sample(min(8, len(news_articles)), random_state=42)

        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            col = cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{row[TITLE_COL]}</div>
                    <div class="card-desc">{row.get(DESC_COL, '')[:200]}...</div>
                    <a href="{row.get(URL_COL, '#')}" target="_blank">Read Full Article ↗</a>
                </div>
                """, unsafe_allow_html=True)

# ------------------------
# NEWS TITLE BASED PAGE
# ------------------------
elif st.session_state.mode == "title":
    st.button("🏠 Back to Home", key="back_home_1", on_click=back_to_home, args=None, kwargs=None, help=None, use_container_width=False)
    st.markdown("### Find Similar Articles")
    selected_article = st.selectbox("Select an article:", news_articles[TITLE_COL].values)
    if st.button("Find Similar Articles"):
        idx = news_articles[news_articles[TITLE_COL] == selected_article].index[0]
        distances = list(enumerate(similarity[idx]))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        sim_rows = [news_articles.iloc[i[0]] for i in distances]
        cols = st.columns(2)
        for i, row in enumerate(sim_rows):
            col = cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{row[TITLE_COL]}</div>
                    <div class="card-desc">{row.get(DESC_COL, '')[:200]}...</div>
                    <a href="{row.get(URL_COL, '#')}" target="_blank">Read Full Article ↗</a>
                </div>
                """, unsafe_allow_html=True)

# ------------------------
# PERSONALIZED PAGE
# ------------------------
elif st.session_state.mode == "personalized":
    st.button("🏠 Back to Home", key="back_home_2", on_click=back_to_home)
    st.markdown("###  Personalized Recommendations")
    users = sorted(user_interest['UserId'].unique())
    selected_user = st.selectbox("👤 Select User ID:", users)
    if st.button(" Get My Recommendations"):
        def hybrid_recommend_for_user(user_id, alpha=0.7, beta=0.3, top_k=6):
            user_data = user_ratings[user_ratings['UserId'] == user_id]
            if user_data.empty:
                return pd.DataFrame()
            all_scores = np.zeros(len(news_articles))
            time_max = user_data['Time Spent (seconds)'].max() or 1
            for _, r in user_data.iterrows():
                title = r['Title']
                rating = r.get('Ratings', 0)
                time_spent = r.get('Time Spent (seconds)', 0)
                if title in news_articles[TITLE_COL].values:
                    idx = news_articles[news_articles[TITLE_COL] == title].index[0]
                    weight = (alpha * rating) + (beta * (time_spent / time_max))
                    sim_vec = similarity[idx]
                    if len(sim_vec) == len(news_articles):
                        all_scores += sim_vec * weight
            read_titles = set(user_data['Title'].values)
            ranked = np.argsort(all_scores)[::-1]
            picks = []
            for i in ranked:
                if len(picks) >= top_k:
                    break
                candidate = news_articles.iloc[i][TITLE_COL]
                if candidate not in read_titles:
                    picks.append(i)
            return news_articles.iloc[picks]

        recs = hybrid_recommend_for_user(selected_user)
        if recs.empty:
            st.info("No personalized recommendations available.")
        else:
            cols = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows()):
                col = cols[i % 2]
                with col:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{row[TITLE_COL]}</div>
                        <div class="card-desc">{row.get(DESC_COL, '')[:200]}...</div>
                        <a href="{row.get(URL_COL, '#')}" target="_blank">Read Full Article ↗</a>
                    </div>
                    """, unsafe_allow_html=True)
