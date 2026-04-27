import os, glob, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss


# ------ 1) Load & parse docs ------
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path):
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def normalize_filename(name):
    return name.strip().lower()

def load_metadata(path="docs/metadata.csv"):
    df = pd.read_csv(path)
    df["filename"] = df["filename"].apply(normalize_filename)
    return df.set_index("filename").to_dict(orient="index")

def load_docs(folder="docs"):
    texts = []
    for fp in glob.glob(os.path.join(folder, "*")):
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext == ".txt":
                content = read_txt(fp)
            elif ext == ".pdf":
                content = read_pdf(fp)
            elif ext == ".docx":
                content = read_docx(fp)
            else:
                continue
            if content and content.strip():
                texts.append((os.path.basename(fp), content))
        except Exception as e:
            print(f"Skip {fp}: {e}")
    return texts

# ------ 2) Cleaning & chunking ------
def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()

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

def build_chunks(docs, metadata):
    rows = []
    cid = 0
    DEBUG = False
    missing_meta = 0

    for src, content in docs:
        ctext = clean_text(content)

        # ambil metadata berdasarkan nama file
        meta = metadata.get(normalize_filename(src), {})
        
        if not meta:
            missing_meta += 1
            continue

        if DEBUG:
            print("DEBUG SRC:", src)
            print("DEBUG META:", meta)  

        for ch in chunk_text(ctext, max_tokens=180, overlap=30):
            if len(ch) > 0:
                rows.append({
                    "chunk_id": cid,
                    "source": src,
                    "text": ch,

                    # 🔥 metadata masuk sini
                    "division": meta.get("division"),
                    "project": meta.get("project"),
                    "country": meta.get("country"),
                    "city": meta.get("city"),
                    "year": meta.get("year"),
                    "document_type": meta.get("document_type"),
                    "access_level": meta.get("access_level"),
                    "owner": meta.get("owner"),
                })
                cid += 1
    
    return pd.DataFrame(rows)

# ------ 3) Build index ------
def build_index(df, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="data/index.faiss", chunks_path="data/chunks.parquet"):
    model = SentenceTransformer(model_name)
    emb = model.encode(df["text"].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity (karena normalized)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, index_path)
    df.to_parquet(chunks_path, index=False)
    print(f"Saved {len(df)} chunks → {index_path}, {chunks_path}")

# ------ 4) Search ------
def search(query, top_k=5, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="index.faiss", chunks_path="chunks.parquet"):
    model = SentenceTransformer(model_name)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.read_index(index_path)
    D, I = index.search(qv.astype(np.float32), top_k)
    df = pd.read_parquet(chunks_path)
    results = df.iloc[I[0]].copy()
    results["score"] = D[0]
    return results[["score", "source", "text"]]

# ------ 4b) Summarization helper ------
def summarize_passages(passages, max_words=120, model_name="sshleifer/distilbart-cnn-12-6"):
    from transformers import pipeline
    # gabungkan potongan (batasi panjang agar model tidak kelebihan)
    joined = "\n".join(passages)
    # kalau terlalu panjang, potong menjadi beberapa bagian lalu rangkum per-bagian
    if len(joined) > 6000:
        chunks = []
        text = joined.split("\n")
        buf = ""
        for line in text:
            if len(buf) + len(line) < 4000:
                buf += line + "\n"
            else:
                chunks.append(buf); buf = line + "\n"
        if buf: chunks.append(buf)
    else:
        chunks = [joined]

    summarizer = pipeline("summarization", model=model_name)
    partial = []
    for c in chunks:
        out = summarizer(c, max_length=max_words, min_length=max(30, max_words//3), do_sample=False)[0]["summary_text"]
        partial.append(out)
    # meta-summary jika lebih dari satu
    if len(partial) > 1:
        out = summarizer("\n".join(partial), max_length=max_words, min_length=max(30, max_words//3), do_sample=False)[0]["summary_text"]
        return out
    return partial[0]


# ------ 5) CLI Mode ------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "query", "answer"], required=True)
    parser.add_argument("--q", type=str, default="")
    args = parser.parse_args()

    if args.mode == "build":
        metadata = load_metadata("docs/metadata.csv")
        docs = load_docs("docs")
        if not docs:
            raise SystemExit("Folder docs kosong. Taruh file .txt/.pdf/.docx di folder docs/")
        df = build_chunks(docs, metadata)
        build_index(df)
    else:
        if not args.q:
            raise SystemExit("Gunakan --q 'pertanyaan anda'")
        out = search(args.q, top_k=5)
        pd.set_option("display.max_colwidth", 160)
        print(out.to_string(index=False))

elif args.mode == "answer":
    if not args.q:
        raise SystemExit("Gunakan --q 'pertanyaan anda'")
    # ambil top-k (misal 6) lalu ringkas
    df = search(args.q, top_k=6)
    passages = df["text"].tolist()
    answer = summarize_passages(passages, max_words=140)
    # tampilkan
    print("\n=== Jawaban Singkat ===\n")
    print(answer)
    print("\n=== Sumber (top) ===")
    for src in df["source"].unique()[:5]:
        print("-", src)
