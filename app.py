"""
app.py — Travel Activity Recommender
=====================================
Streamlit UI: upload up to 5 travel photos → Gemini labels them →
TF-IDF recommender suggests top-10 Texas activities.

Run:
    pip install streamlit google-genai pillow pillow-heif scikit-learn pandas
    streamlit run app.py
"""

import re
import ast
import io
import base64
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ITEMS_CSV   = "things_to_do_all_with_local_with_labels.csv"
GROQ_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"   # free vision model on Groq
TOP_N       = 10
MAX_PHOTOS  = 5

LABEL_PROMPT = """Look at this travel photo and return ONLY a comma-separated
list of 10-15 descriptive visual labels. No explanation, just labels.

Focus on: type of activity, setting (indoor/outdoor, urban/nature),
objects visible, mood, type of place, architecture, scenery.

Example:
hiking, mountain trail, forest, outdoor adventure, scenic view, nature,
friends group, autumn foliage, rocky terrain, sunny day"""


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def parse_labels(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            return " ".join(map(str, items))
        except Exception:
            pass
    return s


@st.cache_data
def load_items(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = next(
        (c for c in df.columns if c.lower() in ("labels_openai", "labels")), None
    )
    title_text = df["title"].fillna("").map(normalize)
    label_text = (
        df[label_col].fillna("").apply(parse_labels).map(normalize)
        if label_col
        else pd.Series([""] * len(df))
    )
    df["_text"] = (title_text + " " + label_text).str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def image_to_pil(uploaded_file) -> Image.Image:
    data = uploaded_file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def label_photo(client, img: Image.Image) -> str:
    try:
        b64 = pil_to_base64(img)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": LABEL_PROMPT},
                ],
            }],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[error: {e}]"


def build_user_profile(all_labels: list[str]) -> str:
    from collections import Counter
    tokens = []
    for label_str in all_labels:
        phrases = [p.strip().lower() for p in label_str.split(",") if p.strip()]
        tokens.extend(phrases)
    counts = Counter(tokens)
    weighted = []
    for token, count in counts.items():
        weighted.extend([token] * count)
    return " ".join(weighted)


def recommend(user_text: str, df_items: pd.DataFrame, top_n: int = TOP_N):
    all_texts = [user_text] + df_items["_text"].tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=1,
        sublinear_tf=True,
    )
    X         = vectorizer.fit_transform(all_texts)
    user_vec  = X[0]
    items_mat = X[1:]
    scores    = cosine_similarity(user_vec, items_mat)[0]
    top_idx   = np.argsort(-scores)[:top_n]

    feat    = np.array(vectorizer.get_feature_names_out())
    results = []
    for j in top_idx:
        p_idx = set(user_vec.indices)
        i_idx = set(items_mat[j].indices)
        inter = np.array(sorted(p_idx & i_idx))
        if inter.size > 0:
            contrib = (
                user_vec[:, inter].toarray().ravel()
                * items_mat[j][:, inter].toarray().ravel()
            )
            order = np.argsort(-contrib)[:5]
            why   = ", ".join(feat[inter[order]])
        else:
            why = ""

        row = df_items.iloc[j]
        results.append({
            "title":    row.get("title", ""),
            "category": row.get("category", ""),
            "url":      row.get("detail_url", ""),
            "score":    round(float(scores[j]), 4),
            "why":      why,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Texas Activity Recommender",
    page_icon="🤠",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com",
    )
    st.markdown(
        "**Get a free key (no credit card):**\n"
        "1. Go to [console.groq.com](https://console.groq.com)\n"
        "2. Sign up with Google or email\n"
        "3. Go to **API Keys** → **Create API key**\n"
        "4. Paste it above"
    )
    st.divider()
    st.caption(f"Vision model: `llama-4-scout` via Groq (free)")

    st.divider()
    st.subheader("ℹ️ How it works")
    st.markdown(
        """
**Step 1 — Vision labeling**
Your photos are sent to **LLaMA 4 Vision** (via Groq), a multimodal AI model.
It reads each image and returns 10–15 descriptive labels like
*"beach, sunset, outdoor dining, tropical vibe"*.

**Step 2 — User profile**
All labels from your photos are merged into a single text "profile"
that represents your travel preferences. Labels that appear in multiple
photos are given more weight.

**Step 3 — TF-IDF matching**
Your profile is compared against **500+ Texas activities** using
**TF-IDF + cosine similarity** — the same algorithm used in search engines.
Each activity has been pre-labeled with keywords describing what it offers.

**Step 4 — Recommendations**
The activities with the highest similarity score are returned, along with
the exact keywords that drove each match.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤠 Texas Activity Recommender")
st.markdown(
    "Upload up to **5 travel photos** and we'll recommend Texas activities "
    "that match your vibe — powered by **Gemini Vision** + **TF-IDF**."
)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DIAGRAM  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Under the hood")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
<div style="text-align:center; padding:12px; background:#1e3a5f;
     border-radius:10px; color:white;">
<div style="font-size:2rem">📸</div>
<b>1. Upload</b><br>
<small>You upload up to 5 travel photos</small>
</div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
<div style="text-align:center; padding:12px; background:#1e3a5f;
     border-radius:10px; color:white;">
<div style="font-size:2rem">🤖</div>
<b>2. Vision AI</b><br>
<small>LLaMA 4 reads each photo and extracts 10–15 labels</small>
</div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
<div style="text-align:center; padding:12px; background:#1e3a5f;
     border-radius:10px; color:white;">
<div style="font-size:2rem">📊</div>
<b>3. TF-IDF</b><br>
<small>Your labels are matched against 500+ Texas activities</small>
</div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
<div style="text-align:center; padding:12px; background:#1e3a5f;
     border-radius:10px; color:white;">
<div style="font-size:2rem">🌟</div>
<b>4. Results</b><br>
<small>Top 10 matching activities with similarity scores</small>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ACTIVITIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_items = load_items(ITEMS_CSV)
except FileNotFoundError:
    st.error(
        f"❌ `{ITEMS_CSV}` not found. "
        "Make sure it's in the same folder as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PHOTO UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📸 Upload your travel photos")
st.caption(
    "Photos you've taken on previous trips — beaches, hikes, restaurants, "
    "museums, whatever you've enjoyed. The more variety the better!"
)

uploaded = st.file_uploader(
    f"Choose up to {MAX_PHOTOS} photos  (JPG, PNG, WEBP, HEIC)",
    type=["jpg", "jpeg", "png", "webp", "heic"],
    accept_multiple_files=True,
)

if uploaded and len(uploaded) > MAX_PHOTOS:
    st.warning(f"⚠️ Only the first {MAX_PHOTOS} photos will be used.")
    uploaded = uploaded[:MAX_PHOTOS]

if uploaded:
    cols = st.columns(len(uploaded))
    for col, f in zip(cols, uploaded):
        f.seek(0)
        img = image_to_pil(f)
        col.image(img, use_container_width=True, caption=f.name)

# ─────────────────────────────────────────────────────────────────────────────
# CTA BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
go = st.button(
    "🔍 Get My Recommendations",
    type="primary",
    disabled=not (uploaded and api_key),
    use_container_width=True,
)

if not api_key:
    st.info("👈 Paste your Groq API key in the sidebar to get started.")
elif not uploaded:
    st.info("☝️ Upload at least one travel photo above.")

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if go:
    if not GROQ_OK:
        st.error(
            "`groq` not installed. "
            "Run: `pip install groq`"
        )
        st.stop()

    client = Groq(api_key=api_key)

    # ── Step 1: Label photos ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 1 — 🤖 Analyzing your photos with LLaMA Vision")
    st.caption(
        "Each photo is sent to LLaMA 4 Vision (via Groq) which identifies the scene, "
        "activities, objects, mood and setting."
    )

    all_labels  = []
    label_rows  = []
    photo_cols  = st.columns(len(uploaded))
    progress    = st.progress(0, text="Sending photos to Gemini Vision…")

    for i, f in enumerate(uploaded):
        f.seek(0)
        img = image_to_pil(f)

        with st.spinner(f"Labeling photo {i+1}/{len(uploaded)}: {f.name}"):
            labels = label_photo(client, img)

        all_labels.append(labels)
        label_rows.append({"photo": f.name, "labels": labels})

        # Show labeled photo immediately
        with photo_cols[i]:
            f.seek(0)
            photo_cols[i].image(image_to_pil(f), use_container_width=True)
            tag_list = [t.strip() for t in labels.split(",") if t.strip()][:6]
            photo_cols[i].caption("🏷️ " + " · ".join(tag_list))

        progress.progress(
            (i + 1) / len(uploaded),
            text=f"Photo {i+1}/{len(uploaded)} analyzed ✓"
        )
        if i < len(uploaded) - 1:
            time.sleep(5)

    progress.empty()

    with st.expander("📋 See all extracted labels", expanded=False):
        for row in label_rows:
            st.markdown(f"**{row['photo']}**")
            st.caption(row["labels"])

    # ── Step 2: Build profile ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 2 — 📊 Building your travel profile")
    st.caption(
        "All labels from your photos are merged into one text profile. "
        "Keywords that appear across multiple photos carry more weight."
    )

    user_profile = build_user_profile(all_labels)

    from collections import Counter
    all_tokens = [t.strip().lower() for lbl in all_labels for t in lbl.split(",") if t.strip()]
    top_tokens = Counter(all_tokens).most_common(12)

    if top_tokens:
        tag_html = " ".join(
            f'<span style="background:#1e3a5f; color:white; padding:4px 10px; '
            f'border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem;">'
            f'{"⭐ " if count > 1 else ""}{token} ({count})</span>'
            for token, count in top_tokens
        )
        st.markdown(f"**Your top travel keywords:**", unsafe_allow_html=False)
        st.markdown(tag_html, unsafe_allow_html=True)

    # ── Step 3: TF-IDF matching ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 3 — 🔎 Matching against Texas activities")
    st.caption(
        f"TF-IDF converts your profile and each of the {len(df_items):,} activities "
        "into vectors, then cosine similarity ranks how close they are."
    )

    with st.spinner("Running TF-IDF similarity…"):
        recs = recommend(user_profile, df_items, top_n=TOP_N)

    # ── Step 4: Results ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🌟 Your Top {TOP_N} Texas Activities")

    for i, rec in enumerate(recs, start=1):
        with st.container():
            left, right = st.columns([0.07, 0.93])
            left.markdown(
                f"<div style='font-size:1.6rem; font-weight:bold; "
                f"color:#d4a017; text-align:center;'>{i}</div>",
                unsafe_allow_html=True,
            )
            with right:
                title = rec["title"]
                url   = rec["url"]
                if url and str(url).startswith("http"):
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")

                meta_parts = []
                if rec["category"]:
                    meta_parts.append(f"📂 {rec['category']}")

                # Score bar
                score_pct = min(int(rec["score"] * 500), 100)
                meta_parts.append(f"📊 similarity: `{rec['score']}`")

                if rec["why"]:
                    meta_parts.append(f"💡 matched on: *{rec['why']}*")

                st.caption("  ·  ".join(meta_parts))
            st.divider()

    st.success(
        f"✅ Done! Analyzed {len(uploaded)} photo(s) → "
        f"matched against {len(df_items):,} Texas activities."
    )
