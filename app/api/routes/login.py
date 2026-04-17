# app/api/routes/login.py
"""
Endpoints for User Authentication (Login, Token, Password Recovery)
"""
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm

from app import crud
from app.api.deps import CurrentUser, SessionDep, get_current_active_superuser
from app.core import security
from app.core.config import settings
from app.models import Token, UserPublic, UserUpdate
from app.models.schemas import Message as GenericMessage, NewPassword

# Import các hàm tiện ích gửi email (nếu có file app/utils.py)
# Nếu bạn chưa tạo file utils.py, hãy comment khối import này để tránh lỗi
try:
    from app.utils import (
        generate_password_reset_token,
        generate_reset_password_email,
        send_email,
        verify_password_reset_token,
    )
    HAS_EMAIL_UTILS = True
except ImportError:
    HAS_EMAIL_UTILS = False

router = APIRouter(tags=["login"])


@router.post("/login/access-token")
def login_access_token(
    session: SessionDep, 
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 compatible token login, get an access token for future requests.
    Dùng để lấy token xác thực người dùng.
    """
    # Xác thực email và password
    user = crud.authenticate(
        session=session, 
        email=form_data.username,  # OAuth2PasswordRequestForm dùng field 'username' cho email
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Tạo access token có thời hạn
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    return Token(
        access_token=security.create_access_token(
            subject=user.id, 
            expires_delta=access_token_expires
        ),
        token_type="bearer",
    )


@router.post("/login/test-token", response_model=UserPublic)
def test_token(current_user: CurrentUser) -> Any:
    """
    Test access token.
    Endpoint này yêu cầu Auth (Bearer Token). Nếu token hợp lệ, sẽ trả về thông tin user.
    """
    return current_user


@router.post("/password-recovery/{email}")
def recover_password(email: str, session: SessionDep) -> GenericMessage:
    """
    Password Recovery
    Gửi email chứa link reset mật khẩu.
    """
    if not HAS_EMAIL_UTILS:
        raise HTTPException(status_code=501, detail="Email service not configured")
        
    user = crud.get_user_by_email(session=session, email=email)

    # Luôn trả về message giống nhau để tránh việc đoán biết email có tồn tại hay không
    if user:
        password_reset_token = generate_password_reset_token(email=email)
        email_data = generate_reset_password_email(
            email_to=user.email, email=email, token=password_reset_token
        )
        send_email(
            email_to=user.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
        
    return GenericMessage(
        message="If that email is registered, we sent a password recovery link"
    )


@router.post("/reset-password/")
def reset_password(session: SessionDep, body: NewPassword) -> GenericMessage:
    """
    Reset password
    Cập nhật mật khẩu mới khi người dùng cung cấp token hợp lệ.
    """
    if not HAS_EMAIL_UTILS:
        raise HTTPException(status_code=501, detail="Email service not configured")
        
    email = verify_password_reset_token(token=body.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token")
        
    user = crud.get_user_by_email(session=session, email=email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    # Cập nhật mật khẩu mới (đã được hash trong crud.update_user)
    user_in_update = UserUpdate(password=body.new_password)
    crud.update_user(
        session=session,
        db_user=user,
        user_in=user_in_update,
    )
    return GenericMessage(message="Password updated successfully")


@router.post(
    "/password-recovery-html-content/{email}",
    dependencies=[Depends(get_current_active_superuser)],
    response_class=HTMLResponse,
)
def recover_password_html_content(email: str, session: SessionDep) -> Any:
    """
    HTML Content for Password Recovery (Dùng để test email template)
    Chỉ dành cho Superuser.
    """
    if not HAS_EMAIL_UTILS:
        raise HTTPException(status_code=501, detail="Email service not configured")
        
    user = crud.get_user_by_email(session=session, email=email)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system.",
        )
        
    password_reset_token = generate_password_reset_token(email=email)
    email_data = generate_reset_password_email(
        email_to=user.email, email=email, token=password_reset_token
    )

    return HTMLResponse(
        content=email_data.html_content, headers={"subject:": email_data.subject}
    )