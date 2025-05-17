from pydantic import BaseModel
from typing import Dict, Any, List

class QueryResponse(BaseModel):
    question: str
    answer: str
    evaluation: Dict[str, float]
    metrics: Dict[str, Any]

class DocumentResponse(BaseModel):
    status: str
    document_id: str
    metadata: Dict[str, Any]
