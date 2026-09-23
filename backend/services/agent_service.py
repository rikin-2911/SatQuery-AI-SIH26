## Here the langgraph agent is initiated for user query
import re
import json
from typing import Any
from agent.graph import build_satgraph


# helper function for JSON handling at frontend
def parse_model_result(result: Any) -> Any:
    """
    Convert the specialist result into a Python object when possible.

    Handles:
    - already parsed dictionaries/lists
    - normal JSON strings
    - ```json ... ``` fenced strings
    """

    if isinstance(result, (dict, list)):
        return result

    if not isinstance(result, str):
        return result

    text = result.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return result

# Confidence Extraction
def extract_confidence(task_type: str | None, data: Any) -> float | None:
    """
    Extract a representative confidence from the structured
    specialist output.

    Confidence is taken from model-produced evidence and should
    be treated as a model confidence score, not a calibrated probability.
    """

    if not isinstance(data, dict):
        return None

    candidates: list[float] = []

    def collect_from_items(items):
        if not isinstance(items, list):
            return

        for item in items:
            if not isinstance(item, dict):
                continue

            value = item.get("confidence")

            if isinstance(value, (int, float)):
                if 0.0 <= float(value) <= 1.0:
                    candidates.append(float(value))

    if task_type == "grounding":
        collect_from_items(data.get("detections"))

    elif task_type == "change":
        collect_from_items(data.get("changes"))

    elif task_type == "sar":
        collect_from_items(data.get("observations"))

        # Also support a top-level confidence if  SAR output
        # provides one.
        value = data.get("confidence")

        if isinstance(value, (int, float)):
            if 0.0 <= float(value) <= 1.0:
                candidates.append(float(value))

    if not candidates:
        return None

    return max(candidates)

# Human readable answer at UI
def build_answer(task_type: str | None, data: Any) -> str:
    """
    Convert structured specialist output into a concise human-readable
    answer for the frontend.
    """

    if not isinstance(data, dict):
        return str(data)

    # Grounding
    if task_type == "grounding":

        detections = data.get("detections", [])

        found = [
            item
            for item in detections
            if isinstance(item, dict)
            and item.get("found") is True
        ]

        if not found:
            return "No reliable target feature was detected in the image."

        answers = []

        for item in found:
            feature = item.get("feature", "Target feature")
            description = item.get("description", "")

            if description:
                answers.append(
                    f"**{feature} detected.** {description}"
                )
            else:
                answers.append(
                    f"**{feature} detected.**"
                )

        return "\n\n".join(answers)

    # SAR
    if task_type == "sar":

        answer = data.get("answer")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        observations = data.get("observations", [])

        if observations:
            parts = []

            for item in observations:
                if not isinstance(item, dict):
                    continue

                feature = item.get("feature", "")
                description = item.get("description", "")

                if feature and description:
                    parts.append(
                        f"**{feature}:** {description}"
                    )
                elif description:
                    parts.append(description)

            if parts:
                return "\n\n".join(parts)

        return "SAR analysis completed, but no textual summary was returned."

    
    # CHANGE
    if task_type == "change":

        changes = data.get("changes", [])

        if not changes:
            return "No significant changes were detected between the two images."

        parts = []

        for item in changes:
            if not isinstance(item, dict):
                continue

            description = (
                item.get("description")
                or item.get("change")
                or item.get("summary")
            )

            if description:
                parts.append(str(description))

        if parts:
            return "\n\n".join(parts)

        return "Changes were detected, but no textual description was returned."

    return str(data)


## Agent Class for calling Laggraph agents
class AgentService:

    def __init__(self, router_llm):

        self.router_llm = router_llm
        # build the graph
        self.graph = build_satgraph(self.router_llm)


    # graph agent calling 
    def analyze(self, query: str, image_paths: list[str]):
        state = {
            "query": query,
            "image_paths": image_paths,
            "task_type": None,
            "answer": None,
            "raw_result": None,
            "result": None,
            "model_used": None,
            "confidence": None,
            "execution_trace": [],
        }

        graph_result = self.graph.invoke(state)

        task_type = graph_result.get("task_type")
        raw_result_string = graph_result.get("result")

        # Parse specialist output
        parsed_result = parse_model_result(raw_result_string)

        # Generate human-readable answer
        answer = build_answer(
            task_type,
            parsed_result,
        )

        # Extract confidence
        confidence = extract_confidence(
            task_type,
            parsed_result,
        )

        return {
            **graph_result,

            "answer": answer,

            "raw_result": parsed_result,

            "confidence": confidence,
        }