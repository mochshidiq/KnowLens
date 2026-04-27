cd D:\KnowLens
pertama kali :
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

.\.venv\Scripts\activate
pip install -r requirements.txt
python build_and_query.py atau python build_and_query.py --mode build
python api.py
uvicorn app.api:app --reload --port 8000
http://127.0.0.1:8000/docs → untuk Swagger UI (tes endpoint).

http://127.0.0.1:8000/query?text=apa+itu+KM → contoh query.