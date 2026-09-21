## Router prompt for guiding the agent for choosing the correct node based on User Query and Input
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


## NODE
def router_node(state: dict, llm) -> dict:
    query = state["query"]
    image_paths = state["image_paths"]

    image_count = len(image_paths)

    user_input = f"""
          QUERY:
          {query}

          IMAGE COUNT:
          {image_count}

          IMAGE PATHS:
          {image_paths}
        """

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ])

    # Parse the LLM JSON response
    import json

    try:
        routing = json.loads(response.content)
        route = routing.get("route")
        reason = routing.get("reason", "")
    except Exception:
        route = None
        reason = "Router returned invalid JSON."

    # Safety validation
    if route not in {"change", "sar", "grounding"}:
        return {
            **state,
            "task_type": None,
            "execution_trace": state.get("execution_trace", []) + [
                "Router failed: invalid route."
            ]
        }

    return {
        **state,
        "task_type": route,
        "execution_trace": state.get("execution_trace", []) + [
            f"Router selected: {route}",
            f"Reason: {reason}"
        ]
    }