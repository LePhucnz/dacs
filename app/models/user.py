# app/models/user.py
from typing import Optional, List
from sqlmodel import SQLModel, Field
from pydantic import EmailStr, validator
import uuid
from datetime import datetime

# ==================== DATABASE MODEL ====================
class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = {"extend_existing": True}
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    full_name: Optional[str] = Field(default=None, max_length=255)
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)


# ==================== PYDANTIC SCHEMAS ====================

class UserBase(SQLModel):
    email: EmailStr = Field(max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=255)
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=40)
    
    @validator('email')
    def email_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Email cannot be empty')
        return v.strip().lower()


class UserUpdate(SQLModel):
    email: Optional[EmailStr] = Field(default=None, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, min_length=8, max_length=40)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserUpdateMe(SQLModel):
    """Schema để user tự cập nhật thông tin cá nhân"""
    email: Optional[EmailStr] = Field(default=None, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, min_length=8, max_length=40)


class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class UsersPublic(SQLModel):
    """Pagination response for users list"""
    data: List[UserPublic]
    count: int


# ==================== TOKEN SCHEMAS ====================

class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(SQLModel):
    sub: Optional[str] = None
    exp: Optional[int] = None