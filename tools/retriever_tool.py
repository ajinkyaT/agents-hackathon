from langchain.tools.retriever import create_retriever_tool
from utils.ingest_data import VectorDB

class RetrieverTool:
    def __init__(self, vector_db: VectorDB, name: str, description: str):
        self.vector_db = vector_db
        self.name = name
        self.description = description

    def get_tool(self):
        return create_retriever_tool(
            self.vector_db.retriever,
            self.name,
            self.description
        )

# # Example usage
# power_weeder_vector_db = VectorDB(file_paths=["../data/power_weeder.csv"])
# power_weeder_tool = RetrieverTool(power_weeder_vector_db, "power_weeder_retriever", "Search and retrieve information about power weeders")

# attachments_vector_db = VectorDB(file_paths=["../data/attachments.csv"])
# attachments_tool = RetrieverTool(attachments_vector_db, "attachments_info_retriever", "Search and retrieve information about different attachment types and their purpose")

# tools = [power_weeder_tool.get_tool(), attachments_tool.get_tool()]

# from langgraph.prebuilt import ToolExecutor
# tool_executor = ToolExecutor(tools)
