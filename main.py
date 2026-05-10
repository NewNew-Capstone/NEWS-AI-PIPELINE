from fastapi import FastAPI
from dotenv import load_dotenv
import logging

from content_bc.router import router as content_router
from analysis_bc.router import get_analysis_service, router as analysis_router
from issue_comparison_bc.router import router as issue_comparison_router
from chatbot_bc.router import router as chatbot_router
from knowledge_graph_bc.router import router as knowledge_graph_router

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News AI Pipeline",
    description="AI pipeline for news bias analysis",
    version="0.1.0"
)

app.include_router(content_router)
app.include_router(analysis_router)
app.include_router(issue_comparison_router)
app.include_router(chatbot_router)
app.include_router(knowledge_graph_router)


@app.on_event("startup")
def preload_analysis_service() -> None:
    """분석 모델을 기동 시점에 미리 로딩해 첫 요청 지연/타임아웃을 완화."""
    try:
        get_analysis_service()
        logger.info("AnalysisService preload complete")
    except Exception:
        logger.exception("AnalysisService preload failed")


@app.get("/")
def root():
    return {"message": "News AI Pipeline is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}
