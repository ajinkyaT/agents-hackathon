# from langchain.schema import AIMessage, HumanMessage
from graphs.rag_graph import agentic_rag_graph
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeColors



image_bytes = agentic_rag_graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
with open("graph.png", "wb") as f:
    f.write(image_bytes)