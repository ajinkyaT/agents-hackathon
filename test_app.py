# from langchain.schema import AIMessage, HumanMessage
from graphs.rag_graph import agentic_rag_graph
# from graphs.nodes import tools
# from langgraph.prebuilt import ToolInvocation

# import pprint

# def sample_function():
#     retreiver_tool = tools[0]
#     tool_input = ToolInvocation
#     response = grass_cutter_vector_db.retriever.get_relevant_documents("precautions to be taken")
#     return response

# print(sample_function())
agentic_rag_graph.get_graph().print_ascii()