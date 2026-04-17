# app/crud/message.py
from typing import List, Optional
from sqlmodel import Session, select, func
from app.models.message import Message
from datetime import datetime


def create_message(
    db: Session,
    session_id: str,
    user_message: str,
    bot_response: str,
    intent: Optional[str] = None,
    intent_confidence: Optional[float] = None,
    sentiment: Optional[str] = None,
    sentiment_scores: Optional[dict] = None,
    risk_level: str = "none",
    risk_score: Optional[float] = None,
    is_crisis: bool = False,
    requires_human_intervention: bool = False,
    user_id: Optional[str] = None,
    # 🔥 THÊM 2 THAM SỐ MỚI CHO CONTEXT WINDOW
    context_used: bool = False,
    context_snippet: Optional[str] = None,
) -> Message:
    """
    Tạo và lưu tin nhắn mới vào PostgreSQL
    """
    db_message = Message(
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
        bot_response=bot_response,
        intent=intent,
        intent_confidence=intent_confidence,
        sentiment=sentiment,
        sentiment_scores=sentiment_scores or {},
        risk_level=risk_level,
        risk_score=risk_score,
        is_crisis=is_crisis,
        requires_human_intervention=requires_human_intervention,
        # 🔥 Lưu context info vào DB
        context_used=context_used,
        context_snippet=context_snippet,
    )
    
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    return db_message


def get_messages_by_session(
    db: Session,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Message]:
    """Lấy danh sách tin nhắn theo session_id (mới nhất trước)"""
    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = db.exec(statement)
    return results.all()


def get_last_n_messages(
    db: Session,
    session_id: str,
    n: int = 5,
) -> List[Message]:
    """
    Lấy n tin nhắn gần nhất của session (để làm context)
    Trả về theo thứ tự thời gian tăng dần (cũ → mới)
    """
    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
    )
    results = db.exec(statement)
    # Đảo ngược để có thứ tự cũ → mới
    return list(reversed(results.all()))


def get_total_messages(db: Session) -> int:
    """Đếm tổng số tin nhắn"""
    statement = select(func.count(Message.id))
    return db.exec(statement).one()


def get_unique_sessions_count(db: Session) -> int:
    """Đếm số phiên hội thoại unique"""
    statement = select(func.count(func.distinct(Message.session_id)))
    return db.exec(statement).one()


def get_crisis_messages_count(
    db: Session,
    since: Optional[datetime] = None,
) -> int:
    """Đếm số tin nhắn crisis"""
    statement = select(func.count(Message.id)).where(Message.is_crisis == True)
    if since:
        statement = statement.where(Message.created_at >= since)
    return db.exec(statement).one()