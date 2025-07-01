# Meta Product Recommendation Engine

A **content‑based** toy recommender that suggests Meta (formerly Facebook) products using simple natural‑language similarity.

## How it works
1. `products.csv` lists sample Meta hardware, software, and ad tools with plain‑text descriptions.
2. `src/main.py` converts descriptions to TF‑IDF vectors (scikit‑learn).
3. Your free‑text query is embedded the same way; cosine similarity surfaces the closest matches.

```
$ pip install -r requirements.txt
$ python src/main.py "vr headset with face tracking" --top 2
```

## Folder layout
```
meta_recommender/
├── products.csv
├── requirements.txt
├── README.md
└── src/
    └── main.py
```

## Extending
* Append more Meta SKUs to `products.csv`.
* Swap TF‑IDF for embeddings (`sentence-transformers/all‑mpnet‑base‑v2`) for richer semantics.
* Wrap the recommender in FastAPI or Streamlit for an interactive demo.

### Inspiration
Mirrors the tidy project style in [`semantic-kernel-nltosql`](https://github.com/zeyduz/semantic-kernel-nltosql) but trimmed to essentials.