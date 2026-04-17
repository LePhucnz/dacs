# app/models/__init__.py
from sqlmodel import SQLModel
from typing import List

# ===== USER =====
from .user import (
    User,
    UserBase,
    UserCreate,
    UserUpdate,
    UserUpdateMe,
    UserPublic,
    UsersPublic,
    Token,
    TokenPayload,
)

# ===== ITEM =====
from .item import (
    Item,
    ItemBase,
    ItemCreate,
    ItemUpdate,
    ItemPublic,
    ItemsPublic,
)

# ===== CHAT MESSAGE (Database model) =====
from .message import Message as ChatMessage

# ===== GENERIC SCHEMAS =====
from .schemas import (
    Message as _GenericMessage,
    NewPassword,
    UpdatePassword,
    RegisterUser,
    UserRegister,
)

# ===== BACKWARD COMPATIBILITY =====
Message = _GenericMessage

# ===== EXPORT ALL =====
__all__ = [
    "SQLModel",
    # User
    "User", "UserBase", "UserCreate", "UserUpdate", "UserUpdateMe",
    "UserPublic", "UsersPublic", "Token", "TokenPayload",
    # Item
    "Item", "ItemBase", "ItemCreate", "ItemUpdate", "ItemPublic", "ItemsPublic",
    # Chat Message
    "ChatMessage",
    # Generic schemas
    "Message", "NewPassword", "UpdatePassword", "RegisterUser", "UserRegister",
]