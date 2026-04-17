# app/models/item.py
from typing import Optional, List
from sqlmodel import SQLModel, Field
import uuid
from datetime import datetime


# ==================== DATABASE MODEL (SQLModel) ====================
class Item(SQLModel, table=True):
    __tablename__ = "item"
    __table_args__ = {"extend_existing": True}
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    title: str = Field(max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    owner_id: uuid.UUID = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)


# ==================== PYDANTIC SCHEMAS (CHO API) ====================
class ItemBase(SQLModel):
    title: str = Field(max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class ItemCreate(ItemBase):
    pass


class ItemUpdate(ItemBase):
    title: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ItemsPublic(SQLModel):
    """Pagination response for items list"""
    data: List[ItemPublic]
    count: int