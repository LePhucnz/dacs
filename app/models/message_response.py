# app/models/message_response.py - Schema response đơn giản
from sqlmodel import SQLModel, Field

class Message(SQLModel):
    """Generic message response schema"""
    message: str = Field(..., description="Message content")