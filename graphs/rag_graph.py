from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from agents.state import AgentState
from graphs.nodes import agent, grade_documents, rewrite, generate, parse_input_question, grade_generation_v_documents_and_question, translate_answer, tools

# Define a new graph
workflow = StateGraph(AgentState)

workflow.add_node("parse_language", parse_input_question)
# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
# retrieve = ToolNode(tools)
workflow.add_node("retrieve", ToolNode(tools))  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)
workflow.add_node("translate_answer", translate_answer)


workflow.set_entry_point("parse_language")
workflow.add_edge("parse_language","agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: 'translate_answer',
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("rewrite", "agent")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "rewrite",
        "useful": 'translate_answer',
        # "not useful": "rewrite",
    },
)
workflow.add_edge('translate_answer', END)

# Compile
agentic_rag_graph = workflow.compile()