import streamlit as st
import pickle
import requests
import re
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Book Recs", page_icon="📚", layout="wide")
st.title("Book Recommendations")
st.caption("Pick a book. Choose a mood. I will introduce it to its literary cousins.")

BASE_DIR = Path(__file__).resolve().parent
newbooks = pickle.load(open(BASE_DIR / "books.pkl", "rb"))
tfidf_matrix = pickle.load(open(BASE_DIR / "tfidf_matrix.pkl", "rb"))

assert tfidf_matrix.shape[0] == len(newbooks), "Mismatch between books.pkl and tfidf_matrix.pkl"

@st.cache_data(show_spinner=False)
def fetch_cover_by_title(title: str):
    url = "https://openlibrary.org/search.json"
    try:
        r = requests.get(url, params={"title": title, "limit": 1}, timeout=5)
        data = r.json()
        if "docs" in data and len(data["docs"]) > 0:
            cover_id = data["docs"][0].get("cover_i")
            if cover_id:
                return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    except Exception:
        pass
    return None

MOODS = {
    "Surprise me 🎲": [],
    "Cozy ☕": ["cozy", "warm", "home", "family", "friendship", "comfort", "heart", "winter"],
    "Adventurous 🧭": ["adventure", "journey", "quest", "travel", "expedition", "survival", "exploration"],
    "Serious 🧐": ["history", "politics", "philosophy", "economics", "society", "ethics", "war", "justice"],
    "Light and funny 😄": ["humor", "funny", "comedy", "satire", "laugh", "witty", "jokes"],
    "Dark and mysterious 🕯️": ["mystery", "crime", "thriller", "dark", "ghost", "haunted", "murder"],
}

_word_boundary_cache = {}

def _contains_word(text: str, word: str) -> bool:
    if word not in _word_boundary_cache:
        _word_boundary_cache[word] = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    return bool(_word_boundary_cache[word].search(text))

def mood_bonus(tags: str, mood_words: list[str]) -> float:
    if not mood_words or not isinstance(tags, str):
        return 0.0
    hits = 0
    for w in mood_words:
        if _contains_word(tags, w):
            hits += 1
    return float(hits)

def recommend(title: str, top_n: int = 10, mood: str = "Surprise me 🎲", mood_weight: float = 0.15, pool: int = 300):
    idxs = newbooks.index[newbooks["Title"].astype(str).str.lower() == title.lower()]
    if len(idxs) == 0:
        return []
    idx = int(idxs[0])

    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1

    pool = min(pool, len(sims) - 1)
    cand_idx = sims.argsort()[-pool:][::-1]

    mood_words = MOODS.get(mood, [])
    scored = []
    for i in cand_idx:
        base = float(sims[i])
        bonus = mood_bonus(newbooks.iloc[i].get("tags", ""), mood_words)
        final = base + mood_weight * bonus
        scored.append((final, i))

    scored.sort(reverse=True)

    chosen = []
    seen = {title.lower()}
    for _, i in scored:
        t = str(newbooks.iloc[i]["Title"])
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        chosen.append(t)
        if len(chosen) == top_n:
            break

    return chosen

st.sidebar.header("🎛️ Controls")
top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)
mood = st.sidebar.selectbox("Reading mood", list(MOODS.keys()))
mood_weight = st.sidebar.slider("Mood strength", 0.0, 0.5, 0.15, 0.05)
show_covers = st.sidebar.checkbox("Show covers", value=True)
snow = st.sidebar.checkbox("Snow mode", value=True)

with st.container():
    option = st.selectbox("Select a book", newbooks["Title"].values)
    recommend_btn = st.button("Recommend", use_container_width=True)

if recommend_btn:
    if snow:
        st.snow()

    recs = recommend(option, top_n=top_n, mood=mood, mood_weight=mood_weight)
    st.subheader("✨ Recommended books")

    for i, title in enumerate(recs, start=1):
        col1, col2 = st.columns([1, 4], vertical_alignment="center")

        with col1:
            if show_covers:
                cover = fetch_cover_by_title(title)
                if cover:
                    st.image(cover, width=120)

        with col2:
            st.markdown(f"### {i}. {title}")
            st.caption(f"Mood: {mood}  |  Strength: {mood_weight}")
