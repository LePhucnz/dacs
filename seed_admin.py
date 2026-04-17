# seed_admin.py - Tạo user admin ban đầu
from sqlmodel import Session, select
from app.core.db import engine
from app.core.security import get_password_hash
from app.models import User

def create_initial_superuser():
    """Tạo superuser mặc định nếu chưa tồn tại"""
    with Session(engine) as session:
        # Kiểm tra xem đã có superuser chưa
        statement = select(User).where(User.is_superuser == True)
        existing = session.exec(statement).first()
        
        if existing:
            print(f"✅ Superuser đã tồn tại: {existing.email}")
            return
        
        # Tạo superuser mới
        admin = User(
            email="admin@dacs.local",
            full_name="Admin User",
            hashed_password=get_password_hash("Admin@123"),  # ⚠️ Đổi mật khẩu cho production!
            is_active=True,
            is_superuser=True,
        )
        
        session.add(admin)
        session.commit()
        print(f"✅ Đã tạo superuser: admin@dacs.local / Admin@123")

if __name__ == "__main__":
    create_initial_superuser()