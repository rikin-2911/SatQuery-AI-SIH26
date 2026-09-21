## LangGraph state that will be pass to the agents or nodes

from typing import TypedDict, Optional, Any

class SatQueryState(TypedDict):

    query: str

    image_paths: list[str]

    # for router node -> selectes the task type
    task_type: Optional[str]
    
    result: Optional[str]

    model_used: Optional[str]

    confidence: Optional[float]

    execution_trace: list[str]