import os
import logging
from pythonjsonlogger import jsonlogger
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch


from dotenv import load_dotenv
load_dotenv(override=True)


# 로깅 설정
# logs 디렉토리 생성 (없으면)
os.makedirs("logs", exist_ok=True)

# 루트 로거 가져오기
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:  # 핸들러가 이미 있으면 추가하지 않음 (로그 중복 기록 방지)
    file_handler = logging.FileHandler("logs/server.json.log")
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)




app = FastAPI(title="Agent_API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health", tags=["Health"])
async def health_check():
    """Elastic Search Health Check"""
    logger.info("Health check requested")
    es = Elasticsearch("http://localhost:9200")
    try:
        if es.ping():
            logger.info("Elasticsearch is healthy")
            return JSONResponse(content={"status": "ok"})
        else:
            logger.error("Elasticsearch ping failed")
            return JSONResponse(status_code=500, content={"status": "error", "message": "Elasticsearch unreachable"})
    except Exception as e:
        logger.exception("Exception during Elasticsearch health check")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# Router 등록
from routers.web import web_search
from routers.wiki import wiki_search
from routers.arxiv import arxiv_search
from routers.refine import refine
from routers.hybrid_search import hybrid_search
from routers.hybrid_search2 import hybrid_search2
from routers.generate import generate

app.include_router(web_search)
app.include_router(wiki_search)
app.include_router(arxiv_search)
app.include_router(refine)
app.include_router(hybrid_search)
app.include_router(hybrid_search2)
app.include_router(generate)


# MCP 서버 생성
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(
    app,
    include_operations=["wiki_search", "safety_search", "tech_paper_search"],
    describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
    describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response
    )

# FastAPI 앱에 MCP 서버 마운트
mcp.mount()


if __name__ == "__main__":
    

    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)