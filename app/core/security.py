# app/core/security.py
"""
Security utilities for authentication and password hashing.
Supports JWT tokens + Bcrypt/Argon2 password hashing.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings
from app.models import TokenPayload

# ===== CONSTANTS =====
ALGORITHM = "HS256"

# ===== PASSWORD HASHING =====
# Hỗ trợ cả Argon2 (mạnh hơn) và Bcrypt (tương thích rộng)
password_hash = PasswordHash(
    (
        Argon2Hasher(),
        BcryptHasher(),
    )
)


def verify_password(
    plain_password: str, 
    hashed_password: str
) -> Tuple[bool, Optional[str]]:
    """
    Xác thực password và tự động nâng cấp hash nếu cần.
    
    Args:
        plain_password: Password người dùng nhập (chưa hash)
        hashed_password: Password đã hash trong database
        
    Returns:
        Tuple[bool, Optional[str]]: 
            - (True, None): Password đúng, hash đã tối ưu
            - (True, new_hash): Password đúng, cần update hash mới
            - (False, None): Password sai
    """
    is_valid, new_hash = password_hash.verify_and_update(
        plain_password, 
        hashed_password
    )
    return is_valid, new_hash


def get_password_hash(password: str) -> str:
    """
    Mã hóa password thành hash (ưu tiên Argon2, fallback Bcrypt).
    
    Args:
        password: Password dạng plain text
        
    Returns:
        str: Password đã hash, sẵn sàng lưu vào database
    """
    return password_hash.hash(password)


def create_access_token(
    subject: str | Any, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Tạo JWT access token cho user.
    
    Args:
        subject: User ID hoặc dict chứa thông tin user (sẽ chuyển thành "sub" claim)
        expires_delta: Thời gian token có hiệu lực (default: 15 phút)
        
    Returns:
        str: JWT token dạng string, dùng cho header Authorization
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=15)
    
    expire = datetime.now(timezone.utc) + expires_delta
    
    # Chuẩn hóa subject thành string
    subject_str = str(subject)
    
    to_encode = {
        "exp": expire,
        "sub": subject_str,
        "iat": datetime.now(timezone.utc),  # Issued at
        "type": "access",  # Token type identifier
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenPayload]:
    """
    Giải mã và xác thực JWT token.
    
    Args:
        token: JWT token từ header Authorization
        
    Returns:
        Optional[TokenPayload]: Payload đã giải mã nếu token hợp lệ, None nếu lỗi
    """
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        # Token đã hết hạn
        return None
    except jwt.InvalidTokenError:
        # Token không hợp lệ (sai signature, format lỗi, v.v.)
        return None
    except Exception:
        # Lỗi khác khi parse payload
        return None


def get_token_subject(token: str) -> Optional[str]:
    """
    Lấy user ID (subject) từ token.
    
    Args:
        token: JWT token
        
    Returns:
        Optional[str]: User ID nếu token hợp lệ, None nếu lỗi
    """
    payload = verify_token(token)
    if payload and payload.sub:
        return payload.sub
    return None