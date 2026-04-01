from fastapi import FastAPI
from dotenv import load_dotenv

from content_bc.router import router as content_router
from analysis_bc.router import router as analysis_router
from issue_comparison_bc.router import router as issue_comparison_router

load_dotenv()

app = FastAPI(
    title="News AI Pipeline",
    description="AI pipeline for news bias analysis",
    version="0.1.0"
)

app.include_router(content_router)
app.include_router(analysis_router)
app.include_router(issue_comparison_router)


@app.get("/")
def root():
    return {"message": "News AI Pipeline is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}