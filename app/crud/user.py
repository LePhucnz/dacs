# app/crud/user.py
from typing import Optional
from sqlmodel import Session, select
from app.core.security import get_password_hash, verify_password
from app.models import User, UserCreate, UserUpdate


def create_user(*, session: Session, user_create: UserCreate) -> User:
    """Tạo user mới với mật khẩu đã hash"""
    db_obj = User.model_validate(
        user_create, 
        update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> User:
    """Cập nhật thông tin user"""
    update_data = user_in.model_dump(exclude_unset=True)
    
    # Nếu có password mới, hash lại
    if update_data.get("password"):
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    # Update các field còn lại
    for field, value in update_data.items():
        setattr(db_user, field, value)
    
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def authenticate(*, session: Session, email: str, password: str) -> Optional[User]:
    """
    Xác thực user: kiểm tra email + password
    Returns: User object nếu thành công, None nếu thất bại
    """
    statement = select(User).where(User.email == email)
    user = session.exec(statement).first()
    
    if not user:
        return None
    
    # ✅ Dùng verify_password từ security module
    is_valid, _ = verify_password(password, user.hashed_password)
    if not is_valid:
        return None
    
    if not user.is_active:
        return None
    
    return user


def get_user_by_email(*, session: Session, email: str) -> Optional[User]:
    """Lấy user theo email"""
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()


def get_user_by_id(*, session: Session, user_id: str) -> Optional[User]:
    """Lấy user theo ID"""
    return session.get(User, user_id)