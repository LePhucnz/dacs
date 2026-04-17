# app/crud/__init__.py
"""
CRUD operations export point.
"""

# ===== MESSAGE CRUD (Chat history) =====
from .message import (
    create_message,
    get_messages_by_session,
    get_last_n_messages,
)

# ===== USER CRUD (Authentication & User management) =====
from .user import (
    create_user,
    update_user,
    authenticate,           # 🔐 Quan trọng: hàm xác thực login
    get_user_by_email,
    get_user_by_id,
)

# ===== EXPORT ALL =====
__all__ = [
    # Message
    "create_message",
    "get_messages_by_session", 
    "get_last_n_messages",
    # User
    "create_user",
    "update_user",
    "authenticate",
    "get_user_by_email",
    "get_user_by_id",
]