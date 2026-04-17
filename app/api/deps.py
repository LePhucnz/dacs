# app/api/deps.py
"""
Dependencies for FastAPI routes.
Provides DB session, authentication, and authorization utilities.
"""

from collections.abc import Generator
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from app.core import security
from app.core.config import settings
from app.core.db import engine
from app.models import TokenPayload, User

# ===== OAUTH2 SCHEME =====
# Endpoint để client lấy access token
reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)


# ===== DATABASE DEPENDENCY =====
def get_db() -> Generator[Session, None, None]:
    """
    Yield database session for each request.
    Automatically commits/rollbacks and closes session.
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()  # Auto-commit nếu không có exception
        except Exception:
            session.rollback()  # Rollback nếu có lỗi
            raise
        finally:
            session.close()  # Luôn đóng session


# Type alias cho DB dependency (dùng trong endpoint signatures)
SessionDep = Annotated[Session, Depends(get_db)]


# ===== AUTHENTICATION DEPENDENCIES =====
def get_current_user(
    session: SessionDep,
    token: str = Depends(reusable_oauth2)
) -> User:
    """
    Xác thực user từ JWT token.
    
    Dependency này được dùng cho các endpoint yêu cầu login.
    
    Args:
        session: Database session (từ get_db)
        token: JWT token từ header Authorization (từ reusable_oauth2)
        
    Returns:
        User: User object đã xác thực
        
    Raises:
        HTTPException 403: Token không hợp lệ hoặc hết hạn
        HTTPException 404: User không tồn tại trong DB
        HTTPException 400: User bị khóa (is_active=False)
    """
    # ✅ Dùng hàm verify_token từ security module (đã xử lý error + timezone)
    payload = security.verify_token(token)
    
    if payload is None or payload.sub is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Lấy user từ database theo ID trong token
    user = session.get(User, payload.sub)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    return user


# Type alias cho authenticated user dependency
CurrentUser = Annotated[User, Depends(get_current_user)]


# ===== AUTHORIZATION DEPENDENCIES =====
def get_current_active_superuser(current_user: CurrentUser) -> User:
    """
    Kiểm tra user có phải superuser không.
    
    Dependency này được dùng cho các endpoint admin-only.
    
    Args:
        current_user: User đã xác thực (từ get_current_user)
        
    Returns:
        User: Same user object nếu là superuser
        
    Raises:
        HTTPException 403: User không phải superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user


# Type alias cho superuser dependency
SuperUserDep = Annotated[User, Depends(get_current_active_superuser)]


# ===== OPTIONAL: Dependency cho public endpoints (không yêu cầu auth) =====
def get_optional_user(
    session: SessionDep,
    token: Optional[str] = Depends(reusable_oauth2)
) -> Optional[User]:
    """
    Lấy user nếu có token hợp lệ, trả về None nếu không có hoặc token lỗi.
    
    Dùng cho endpoints cho phép cả guest và logged-in users.
    """
    if token is None:
        return None
    
    try:
        payload = security.verify_token(token)
        if payload and payload.sub:
            return session.get(User, payload.sub)
    except Exception:
        # Bỏ qua lỗi, trả về None cho guest users
        pass
    
    return None


OptionalUserDep = Annotated[Optional[User], Depends(get_optional_user)]