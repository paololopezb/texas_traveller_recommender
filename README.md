# 🤠 Texas Activity Recommender

Upload travel photos and get personalized Texas activity recommendations — powered by **LLaMA 4 Vision** + **TF-IDF**.

## How it works

1. You upload up to 5 travel photos
2. LLaMA 4 Vision (via Groq) reads each photo and extracts descriptive labels
3. Your labels are combined into a travel preference profile
4. TF-IDF cosine similarity matches your profile against 500+ Texas activities
5. Top 10 recommendations are returned with similarity scores and matched keywords

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/paololopezb/texas_traveller_recommender.git
cd texas_traveller_recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card needed)
3. Go to **API Keys → Create API key**
4. Copy the key (starts with `gsk_...`)

### 4. Run the app
```bash
streamlit run app.py
```

Paste your Groq API key in the sidebar, upload your photos, and click **Get My Recommendations**.

## Data

`things_to_do_all_with_local_with_labels.csv` — scraped from [The Culture Trip](https://theculturetrip.com) and labeled with AI. Contains 500+ Texas activities with titles, categories, URLs and descriptive keywords.
