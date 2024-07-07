from typing import Annotated, Sequence, TypedDict, Optional, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import langcodes



class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    context: str
    generation: str
    img_description: str
    img_tags: List[str]
    lang_code: Annotated[str, langcodes.tag_parser.parse_tag]
    next_step: str
