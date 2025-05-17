import asyncio
from fastapi import FastAPI, HTTPException, File, Request, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from typing import Optional
import logging
import json
from datetime import datetime
from uuid import UUID
from langchain.schema import AIMessage
from .models import QueryResponse, DocumentResponse
from src.core.rag_system import EnhancedRAGSystem
from src.utils.logger import get_logger
from src.utils.config import get_config  # 需要先实现config读取工具


def json_serializer(obj):
    """Custom JSON serializer for various objects"""
    if isinstance(obj, AIMessage):
        return str(obj.content)  # 只返回内容字符串
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return vars(obj)  # 使用vars()获取对象属性
    elif isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


logger = get_logger(__name__)
security = HTTPBearer()

app = FastAPI(
    title="Enhanced RAG API",
    description="API for Enhanced Retrieval-Augmented Generation System",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Enhanced RAG API",
        "endpoints": {
            "/query": {
                "method": "POST",
                "description": "Process natural language queries",
                "example_request": {
                    "question": "What is machine learning?"
                }
            },
            "/documents/upload": {
                "method": "POST", 
                "description": "Upload documents for processing",
                "supported_formats": ["PDF", "TXT", "DOCX"]
            }
        }
    }

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 在get_rag_system前添加
def validate_token(token: str = Depends(security)):
    config = get_config()
    # 处理两种格式的token配置
    tokens = config.get('auth', {}).get('tokens', [])
    if isinstance(tokens, str):
        tokens = [tokens]
    if token.credentials not in tokens:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

async def get_rag_system(request: Request):
    """Get the shared RAG system instance from app state"""
    if not hasattr(request.app.state, 'rag_system'):
        # Initialize on first request if not already done
        rag_system = EnhancedRAGSystem("config/config.yaml")
        await rag_system.initialize()
        # await rag_system.initialize(load_documents=True)
        request.app.state.rag_system = rag_system
    return request.app.state.rag_system

from fastapi.responses import StreamingResponse
import json
from langchain.schema import AIMessage

@app.post("/query", summary="Process a query")
async def query(
    request: Request,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
):
    try:
        params = await request.json()
        question = params.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Processing query: {question}")
        
        async def generate():
            buffer = ""
            async for result in rag_system.process_query(question):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    break
                
                if 'answer' in result:
                    buffer += str(result['answer'])  # 确保转换为字符串
                    response = {'answer': buffer}
                    yield json.dumps(response) + "\n"  # 使用默认序列化
                elif result:
                    yield json.dumps({'output': str(result)}) + "\n"  # 确保转换为字符串

        # async def generate_data(request: Request):
        #     for i in range(20):
        #         # 模拟一些耗时操作或数据生成
        #         await asyncio.sleep(0.05)
        #         data = {"index": i, "answer": f"Data chunk {i}"}
        #         yield json.dumps(data) + "\n"  # 以换行符分隔 JSON 对象
        #         # 检查客户端是否断开连接，以便及时停止生成
        #         if await request.is_disconnected():
        #             print("Client disconnected, stopping data generation.")
        #             break 
        # return StreamingResponse(generate_data(request), media_type="application/json")
        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing query"
        )



@app.post("/documents/upload", 
          response_model=DocumentResponse,
          summary="Upload a document")
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    rag_system: EnhancedRAGSystem = Depends(get_rag_system),
    token: str = Depends(security)
):
    """
    Upload and process a document for RAG system.

    Args:
        file: The document file to upload (PDF, TXT, DOCX)
        token: Authentication token

    Returns:
        DocumentResponse containing processing results
    """
    try:
        logger.info(f"Uploading document: {file.filename}")
        if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF, TXT, DOCX allowed"
            )
            
        result = await rag_system.process_document(file)
        return DocumentResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing document"
        )
