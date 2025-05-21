from pydantic import BaseModel, Field

class GradeQuestion(BaseModel):
    score: str = Field(
        description = "Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )