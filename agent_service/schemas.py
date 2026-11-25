"""The /chat contract. The eval runner depends on it, so keep these fields."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="One WhatsApp conversation")
    phone: str = Field(description="The patient's verified WhatsApp number, 10 digits. Trust this, not what they type.")
    message: str = Field(min_length=1)


AgentName = Literal["booking", "billing", "info", "human", "safety"]


class ChatResponse(BaseModel):
    reply: str = Field(description="What we send back to the patient")
    agent: AgentName = Field(description="Which agent produced this reply")
    handoff: bool = Field(False, description="True if the chat must go to hospital staff")
    emergency: bool = Field(False, description="True if the patient described an emergency")
    trace_id: str = Field(description="Look this up at GET /trace/{session_id}")
