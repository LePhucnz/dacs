# app/models/message.py
from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Column, String, Text, Float, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSON
from sqlmodel import SQLModel, Field


class Message(SQLModel, table=True):
    """Database model for chatbot conversation history"""
    __tablename__ = "messages"  # 👈 2 dấu gạch dưới
    __table_args__ = {"extend_existing": True}  # 👈 2 dấu gạch dưới

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    session_id: str = Field(index=True)
    user_id: Optional[str] = Field(None, index=True)

    user_message: str = Field(sa_column=Column(Text))
    bot_response: str = Field(sa_column=Column(Text))

    # Metadata phân tích
    intent: Optional[str] = Field(None, sa_column=Column(String(length=50)))
    intent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sentiment: Optional[str] = Field(None, sa_column=Column(String(length=30)))
    sentiment_scores: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Risk assessment
    risk_level: str = Field(default="none", sa_column=Column(String(length=20)))
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_crisis: bool = Field(default=False)
    requires_human_intervention: bool = Field(default=False)

    # Context tracking
    context_used: bool = Field(default=False)
    context_snippet: Optional[str] = Field(None, sa_column=Column(Text))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: Optional[datetime] = Field(None)