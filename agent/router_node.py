## Router prompt for guiding the agent for choosing the correct node based on User Query and Input
import json
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

ROUTES = Literal["change", "sar", "grounding"]

SYSTEM_PROMPT = """
You are the routing controller for a remote-sensing analysis system.

Select exactly ONE route for the user's request.

AVAILABLE ROUTES:

- change:
  Use when the task compares TWO images from different times and asks
  about changes, change detection, change description, or change
  localization.

- sar:
  Use when the task requires SAR-specific interpretation, such as
  radar backscatter, SAR land-cover analysis, radar shadow, layover,
  scattering, or SAR-specific reasoning.

- grounding:
  Use when the main task is to locate or spatially identify a feature
  in an image, especially when bounding boxes or spatial localization
  are requested.

ROUTING RULES:

1. TWO images + temporal/change intent → "change"
2. SAR-specific analysis → "sar"
3. Spatial localization/bounding boxes → "grounding"
4. Select exactly ONE route.
5. Choose based on the PRIMARY task requested by the user.
6. Do not infer unsupported capabilities.

IMPORTANT:
A SAR image does NOT automatically mean "sar".
For example, "Locate the water body in this SAR image" → "grounding".

Return ONLY valid JSON:

{
  "route": "change | sar | grounding",
  "reason": "Brief reason."
}
"""

def router_node(state, llm):

    # user input query
    query = state['query']

    # input tiff images counting
    image_paths = state['image_paths']
    img_count = len(image_paths)

    user_input = f"""
    USER QUERY: {query}
    NUMBER OF IMAGES: {img_count}
    IMAGE_PATHS: {image_paths}
    """

    # result from router llm -> llama3.1-8B (probably)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ])

    try:
        routing = json.loads(response.content)

        route = routing.get("route")
        reason = routing.get("reason", "")

    except (json.JSONDecodeError, TypeError):
        route = None
        reason = "Router returned Invalied JSON"

    # Validate route
    valid_routes = {"change", "sar", "grounding"}

    if route not in valid_routes:

      return {
            **state,
            "task_type": state.get("execution_trace", []) + ["Router failed: Invalid Route"]
        }
    
    return {
        **state,
        "task_type": state.get("execution_trace", []) + [f"Router Selected: {route}", f"Router Reason: {reason}"]
    }
