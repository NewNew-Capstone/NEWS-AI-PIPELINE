from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="News AI Pipeline",
    description="AI pipeline for news bias analysis",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "News AI Pipeline is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}