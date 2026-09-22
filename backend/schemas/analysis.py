# Request-Response Schemas for Output and Input of the Data

from pydantic import BaseModel
from typing import Optional, Any

class AnalysisResponse(BaseModel):

    query:str
    task_type: Optional[str] = None
    result: Optional[Any] = None
    model_used: Optional[Any] = None
    confidence: Optional[float] = None
    execution_trace: list[str] = []

