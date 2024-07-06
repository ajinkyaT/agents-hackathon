from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Data model

class ImageTags(BaseModel):
    """Binary score to assess answer addresses question."""
    image_description: str = Field(name="image_description",
        description="Short image description"
    )
    extracted_tags: List[str] = Field(name="list_tags",
        description="Tags present in the image from list ['Wet Grass', 'Thin trees', 'Pavement', 'Farm','Short grass','Thick grass','Dry grass,'stones','Garden','Building']"
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class ParseLangCode(BaseModel):
    """Binary score to assess answer addresses question."""
    translated_question: str = Field(
        description="User question translated in English"
    )
    langcode: str = Field(
        description="ISO language code for language of given query"
    )
