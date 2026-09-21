## Agent Graph building using LangGraph and LangChain
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import START, END, StateGraph
from IPython.display import Image, display


# State of the graph to be shared with all the usable nodes during the execution of agent
from agent.state import SatQueryState

# router node for agentic rounting to the specialist agents as per user query
from agent.router_node import router_node

# specialist models as agent nodes
from specialist_models.change_agent import change_node
from specialist_models.grounding_agent import grounding_node
from specialist_models.sar_agent import sar_node # sar node internally call sar specialist class

# for conditional routing using router node to the specilaist agents
def router_task(state):
    route = state.get("task_type")

    if route not in {"sar", "change", "grounding"}:
        raise ValueError(
            f"Invalid routing value: {route!r}"
        )
    
    return route

## llm for routing 
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

model = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        max_new_tokens=300,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.2,
    )

llm = ChatHuggingFace(llm=model)

def build_satgraph(router_llm):
    # Workflow of the graph
    sat_graph = StateGraph(SatQueryState)

    ## NODES
    sat_graph.add_node("router", lambda state: router_node(state, router_llm))
    sat_graph.add_node("sar_node", sar_node)
    sat_graph.add_node("change_node", change_node)
    sat_graph.add_node("grounding_node", grounding_node)

    ## EDGES
    sat_graph.add_edge(START, "router")
    sat_graph.add_conditional_edges(
        "router",
        router_task,
        {   "sar": "sar_node",
            "change": "change_node",
            "grounding":"grounding_node"
        }
    )
    sat_graph.add_edge("sar_node", END)
    sat_graph.add_edge("change_node", END)
    sat_graph.add_edge("grounding_node", END)


    return sat_graph.compile()


## FINAL TESTING OF WHOLE WORKFLOW....
state = {
    "query": "What is the dominant land-cover type in this SAR image?",

    "image_paths": [
        "/home/rikin/satquery-ai/satquery_sar_test/sample_VV.tif"
    ],

    "task_type": None,
    "result": None,
    "model_used": None,
    "confidence": None,
    "execution_trace": []
}

graph = build_satgraph(llm)

result = graph.invoke(state)
print("Task:", result["task_type"])
print("Result:", result["result"])
print("Model:", result["model_used"])
print("Confidence:", result["confidence"])

print("\nExecution Trace:")

for step in result["execution_trace"]:
    print(" →", step)