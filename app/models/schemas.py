# app/models/schemas.py
from sqlmodel import SQLModel, Field
from pydantic import EmailStr, validator
from typing import Optional


class Message(SQLModel):
    """Generic message response schema"""
    message: str = Field(..., description="Message content")


class NewPassword(SQLModel):
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, max_length=40)
    
    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UpdatePassword(SQLModel):
    current_password: str = Field(..., min_length=1, max_length=40)
    new_password: str = Field(..., min_length=8, max_length=40)


class RegisterUser(SQLModel):
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=40)
    full_name: Optional[str] = Field(default=None, max_length=255)


class UserRegister(RegisterUser):
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False