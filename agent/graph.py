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

## Making nodes for shared state across specialist agents

# for conditional routing using router node to the specilaist agents
def router_task(state):
    route = state.get("task_type")

    if route not in {"sar", "change", "grounding"}:
        raise ValueError(
            f"Invalid routing value: {route!r}"
        )
    
    return route

# 1. SAR
def sar_graph_node(state:SatQueryState):

    result = sar_node(tiff_image=state['image_paths'][0],
                      query=state['query'])

    return {
        **state,
        "result":result,
        "task_type":"sar",
        "model_used":"Qwen3.5-27B + AlignEarth-SAR-ViT-B-16",
        "execution_trace":state['execution_trace'] + ["Executed SAR Analysis"]
    }

# 2. Change
def change_graph_node(state:SatQueryState):
    result = change_node(before_tiff_image=state['image_paths'][0],
                         after_tiff_image=state['image_paths'][1],
                         query=state['query'])

    return {
        **state,
        "result":result,
        "task_type":"change",
        "model_used":"Qwen3.5-27B",
        "execution_trace":state['execution_trace'] + ["Executed Change or Bi-Temporal Analysis"]
    }

# 3. Grounding
def grounding_graph_node(state:SatQueryState):
    result = grounding_node(tiff_image=state['image_paths'][0],
                            query=state['query'])

    return {
        **state,
        "result":result,
        "task_type":"grounding",
        "model_used":"Qwen3.5-27B",
        "execution_trace":state['execution_trace'] + ["Executed Grounding Analysis"]
    }

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
    sat_graph.add_node("sar_node", sar_graph_node)
    sat_graph.add_node("change_node", change_graph_node)
    sat_graph.add_node("grounding_node", grounding_graph_node)

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

sar_state = {
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

gnd_state = {
    "query": "Where is the main water body in this image? Provide its bounding box.",

    "image_paths": [
        "/home/rikin/satquery-ai/satquery_sar_test/sample_VV.tif"
    ],

    "task_type": None,
    "result": None,
    "model_used": None,
    "confidence": None,
    "execution_trace": []
}

change_state = {
    "query": "Explain both the images. Describe the changes between two images if any and what they are?",

    "image_paths": [
        "/home/rikin/satquery-ai/imagery_VH.tif",
        "/home/rikin/satquery-ai/imagery_VV.tif"
    ],

    "task_type": None,
    "result": None,
    "model_used": None,
    "confidence": None,
    "execution_trace": []
}



graph = build_satgraph(llm)

result = graph.invoke(change_state)

print(result)