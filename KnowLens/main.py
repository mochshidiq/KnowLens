from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import app as api_app

app = FastAPI()

# mount API
app.mount("/api", api_app)

# serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
