# api.py
import os, re, glob, threading
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sentence_transformers import SentenceTransformer
import faiss

# ---------- CONFIG ----------
DOCS_DIR = "docs"
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.parquet")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="KnowLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CACHE ----------
_model = None
_index = None
_chunks_df = None

# ---------- USERS ----------
USERS = [
    {"username": "andi", "division": "Research", "project": "KM"},
    {"username": "budi", "division": "IT", "project": "SystemX"},
    {"username": "sari", "division": "HR", "project": "HR_policy"},
]

def get_user(username=None):
    for u in USERS:
        if u["username"] == username:
            return u
    return USERS[0]

# ---------- MODEL ----------
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# ---------- LOAD ----------
def load_index():
    global _index, _chunks_df
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    if os.path.exists(CHUNKS_PATH):
        _chunks_df = pd.read_parquet(CHUNKS_PATH)

# ---------- PARSER ----------
def read_txt(p): 
    return open(p, encoding="utf-8", errors="ignore").read()

def read_pdf(p):
    from pypdf import PdfReader
    r = PdfReader(p)
    return "\n".join((pg.extract_text() or "") for pg in r.pages)

def clean(t): 
    return re.sub(r"\s+", " ", (t or "")).strip()

# 🔥 IMPROVED CHUNKING
def chunk_text(text, max_tokens=180, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- METADATA ----------
def load_metadata():
    path = os.path.join(DOCS_DIR, "metadata.csv")
    if not os.path.exists(path): 
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["filename"] = df["filename"].str.lower().str.strip()
    return df

# ---------- BUILD ----------
def build_index_internal():
    docs = []
    for fp in glob.glob(os.path.join(DOCS_DIR, "*")):
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext == ".txt":
                content = read_txt(fp)
            elif ext == ".pdf":
                content = read_pdf(fp)
            else:
                continue
            docs.append((os.path.basename(fp), clean(content)))
        except:
            pass
    
    if not docs:
        return {"status": "no docs found"}

    metadata_df = load_metadata()
    rows = []
    cid = 0

    for src, text in docs:
        meta = metadata_df[metadata_df["filename"] == src.lower()]
        meta = meta.iloc[0].to_dict() if not meta.empty else {}

        for ch in chunk_text(text):
            rows.append({
                "chunk_id": cid,
                "source": src,
                "text": ch,
                **meta
            })
            cid += 1

    df = pd.DataFrame(rows)

    model = get_model()
    emb = model.encode(df["text"].tolist(), normalize_embeddings=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    df.to_parquet(CHUNKS_PATH, index=False)

    load_index()
    return {"chunks": len(df)}

# ---------- SCORING ----------
def keyword_score(text, words):
    t = text.lower()
    return sum(1 for w in words if f" {w} " in f" {t} ")

def quality_score(text):
    t = text.lower()
    score = 0
    if "daftar pustaka" in t or "references" in t:
        score -= 0.3
    if "adalah" in t or "is" in t:
        score += 0.2
    return score

def filter_access(df, user):
    if "access_level" not in df.columns:
        return df
    return df[
        (df["access_level"] == "public") |
        ((df["access_level"] == "division") & (df["division"] == user["division"])) |
        ((df["access_level"] == "project") & (df["project"] == user["project"]))
    ]

# ---------- SEARCH ----------
@app.get("/search")
def search(q: str, username: str = None, k: int = 5):
    if _index is None:
        load_index()

    user = get_user(username)
    model = get_model()

    # ❌ NO MORE QUERY EXPANSION
    qv = model.encode([q], normalize_embeddings=True)

    D, I = _index.search(qv.astype(np.float32), 30)

    res = _chunks_df.iloc[I[0]].copy()
    res["score"] = D[0]

    # 🔥 FILTER RELEVANCE
    res = res[res["score"] > 0.35]

    res = filter_access(res, user)

    words = set(q.lower().split())

    res["keyword_score"] = res["text"].apply(lambda x: keyword_score(x, words))
    res["quality_score"] = res["text"].apply(quality_score)

    # 🔥 HYBRID SCORING
    res["final_score"] = (
        (0.7 * res["score"]) +
        (0.25 * res["keyword_score"]) +
        (0.05 * res["quality_score"])
    )

    res = res.sort_values(by="final_score", ascending=False).head(k)

    res["snippet"] = res["text"].str[:300]

    return {
        "query": q,
        "user": user,
        "results": res.to_dict(orient="records")
    }

# ---------- ANSWER ----------
@app.get("/answer")
def api_answer(q: str, username: str = None):
    if _index is None:
        load_index()

    user = get_user(username)
    model = get_model()

    qv = model.encode([q], normalize_embeddings=True)
    D, I = _index.search(qv.astype(np.float32), 10)

    res = _chunks_df.iloc[I[0]].copy()
    res = filter_access(res, user)

    # 🔥 ambil hanya chunk relevan
    res = res[D[0] > 0.35]

    passages = [
        t for t in res["text"].tolist()
        if len(t.split()) > 40 and "daftar pustaka" not in t.lower()
    ][:3]

    if not passages:
        return {
            "query": q,
            "answer": "Tidak ditemukan jawaban yang relevan.",
            "sources": []
        }

    answer = " ".join(passages)

    return {
        "query": q,
        "answer": answer,
        "sources": res["source"].unique().tolist()
    }

# ---------- BUILD ----------
@app.post("/build")
def api_build():
    stats = build_index_internal()
    return {"status": "ok", "stats": stats}

# ---------- UPLOAD ----------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        path = os.path.join(DOCS_DIR, f.filename)
        with open(path, "wb") as out:
            out.write(await f.read())
        saved.append(f.filename)

    threading.Thread(target=build_index_internal).start()

    return {"uploaded": saved}