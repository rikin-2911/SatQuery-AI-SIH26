import os
from dotenv import load_dotenv

from fastapi import FastAPI

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from agent.graph import build_satgraph
from backend.services.agent_service import AgentService

# all backend APIs is in the router
from backend.api.routes import router

load_dotenv()

# initialization of router LLM when starting of backend
def create_router_llm():
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

    model = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="text-generation",
            max_new_tokens=300,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.2,
        )

    return ChatHuggingFace(llm=model)


# fastapi app
app = FastAPI(title="SatQuery AI",
    description="Interactive Vision-Language Assistant for Remote Sensing Image Analysis",
    version="0.1.0",
    )

@app.on_event("startup")
def startup_event():

    router_llm = create_router_llm()
    app.state.agent_service = AgentService(router_llm=router_llm)


app.include_router(
    router,
    prefix="/api/v1"
)