## Here the langgraph agent is initiated for user query

from agent.graph import build_satgraph

class AgentService:

    def __init__(self, router_llm):

        self.router_llm = router_llm
        # build the graph
        self.graph = build_satgraph(self.router_llm)


    def analyze(self, query: str, image_paths: list[str]):

        state = {
            "query":query,
            "image_paths": image_paths,
            "task_type": None,
            "result": None,
            "model_used": None,
            "confidence": None,
            "execution_trace": [],
        }

        result = self.graph.invoke(state)

        return result