import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse, os

def load_products(path):
    return pd.read_csv(path)

def build_vectorizer(text_series):
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(text_series)
    return vec, matrix

def recommend(df, vec, matrix, query, top_n=3):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    best_idx = sims.argsort()[::-1][:top_n]
    return df.iloc[best_idx]

def cli():
    ap = argparse.ArgumentParser(description="Meta Product Recommendation Engine")
    ap.add_argument("query", help="Describe what you need (e.g., 'portable vr headset for mixed reality')")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "products.csv"))
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    df = load_products(args.data)
    vec, matrix = build_vectorizer(df['description'])
    top_items = recommend(df, vec, matrix, args.query, args.top)

    print(f"\nTop {args.top} recommendations:")
    for _, row in top_items.iterrows():
        print(f"- {row['name']} ({row['category']}) -> {row['description']}")

if __name__ == "__main__":
    cli()