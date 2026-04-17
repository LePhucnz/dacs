"""
Analytics endpoints for Mental Health Chatbot
Thống kê, trực quan hóa và xuất báo cáo hội thoại
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import io
import csv

from app.risk_detector import RiskLevel

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ==================== RESPONSE MODELS ====================

class AnalyticsSummary(BaseModel):
    total_conversations: int
    total_messages: int
    unique_users: int
    crisis_conversations: int
    avg_sentiment_score: float
    dominant_intent: str
    last_updated: str

class SentimentTrendItem(BaseModel):
    timestamp: str
    sentiment: str
    score: float

class RiskDistribution(BaseModel):
    none: int
    low: int
    medium: int
    high: int
    critical: int

class IntentBreakdown(BaseModel):
    intent: str
    count: int
    percentage: float


# ==================== HELPER: LAZY IMPORT TO AVOID CIRCULAR DEPENDENCY ====================
def get_conversations_store():
    """Import conversation store at runtime to avoid circular imports"""
    from app.api.routes.chat import _conversations
    return _conversations


# ==================== ENDPOINTS ====================

@router.get("/summary", response_model=AnalyticsSummary, summary="Get Overall Analytics")
async def get_analytics_summary():
    """
    Thống kê tổng quan: số hội thoại, tin nhắn, crisis, sentiment trung bình, intent phổ biến nhất
    """
    store = get_conversations_store()
    if not store:
        raise HTTPException(status_code=404, detail="No conversation data available")
    
    total_conv = len(store)
    total_msgs = sum(len(conv.messages) for conv in store.values())
    unique_users = len({conv.user_id for conv in store.values() if conv.user_id})
    crisis_conv = sum(1 for conv in store.values() if conv.metadata.get("crisis_count", 0) > 0)
    
    # Calculate avg sentiment score (simple mapping)
    sentiment_map = {"very_negative": -1.0, "negative": -0.5, "neutral": 0.0, "positive": 0.5, "very_positive": 1.0}
    scores = []
    intents_count = {}
    
    for conv in store.values():
        last_sent = conv.metadata.get("last_sentiment")
        if last_sent:
            scores.append(sentiment_map.get(last_sent, 0.0))
        
        last_intent = conv.metadata.get("last_intent")
        if last_intent:
            intents_count[last_intent] = intents_count.get(last_intent, 0) + 1
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    dominant_intent = max(intents_count, key=intents_count.get) if intents_count else "unknown"
    
    return AnalyticsSummary(
        total_conversations=total_conv,
        total_messages=total_msgs,
        unique_users=unique_users or 0,
        crisis_conversations=crisis_conv,
        avg_sentiment_score=round(avg_score, 3),
        dominant_intent=dominant_intent,
        last_updated=datetime.utcnow().isoformat()
    )


@router.get("/sentiment-trend", response_model=List[SentimentTrendItem], summary="Get Sentiment Trend")
async def get_sentiment_trend(limit: int = 50):
    """
    Lấy xu hướng sentiment theo thời gian (tin nhắn gần nhất)
    """
    store = get_conversations_store()
    trend = []
    
    for conv in store.values():
        for msg in conv.messages[-limit:]:
            if msg.role == "user":
                conv_sent = conv.metadata.get("last_sentiment", "neutral")
                trend.append(SentimentTrendItem(
                    timestamp=msg.timestamp.isoformat(),
                    sentiment=conv_sent,
                    score={"very_negative": -1.0, "negative": -0.5, "neutral": 0.0, "positive": 0.5, "very_positive": 1.0}.get(conv_sent, 0.0)
                ))
    
    trend.sort(key=lambda x: x.timestamp)
    return trend[-limit:]


@router.get("/risk-distribution", response_model=RiskDistribution, summary="Get Risk Level Distribution")
async def get_risk_distribution():
    """
    Phân bố số lượng hội thoại theo mức độ nguy cơ
    """
    store = get_conversations_store()
    dist = {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
    
    for conv in store.values():
        level = conv.metadata.get("last_risk_level", "none")
        if level in dist:
            dist[level] += 1
    
    return RiskDistribution(**dist)


@router.get("/intent-breakdown", response_model=List[IntentBreakdown], summary="Get Intent Breakdown")
async def get_intent_breakdown():
    """
    Thống kê tỷ lệ các intent đã phân loại
    """
    store = get_conversations_store()
    intent_counts = {}
    total = 0
    
    for conv in store.values():
        intent = conv.metadata.get("last_intent")
        if intent:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            total += 1
    
    breakdown = []
    for intent, count in intent_counts.items():
        breakdown.append(IntentBreakdown(
            intent=intent,
            count=count,
            percentage=round((count / total) * 100, 2) if total > 0 else 0.0
        ))
    
    breakdown.sort(key=lambda x: x.count, reverse=True)
    return breakdown


@router.get("/export/csv", summary="Export Conversations to CSV")
async def export_conversations_csv():
    """
    Xuất toàn bộ lịch sử hội thoại ra file CSV
    """
    store = get_conversations_store()
    if not store:
        raise HTTPException(status_code=404, detail="No data to export")
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["conversation_id", "user_id", "role", "content", "timestamp", "risk_level", "sentiment", "intent"])
    
    for conv in store.values():
        risk = conv.metadata.get("last_risk_level", "unknown")
        sent = conv.metadata.get("last_sentiment", "unknown")
        intent = conv.metadata.get("last_intent", "unknown")
        
        for msg in conv.messages:
            writer.writerow([
                conv.conversation_id,
                conv.user_id or "anonymous",
                msg.role,
                msg.content.replace("\n", " "),
                msg.timestamp.isoformat(),
                risk,
                sent,
                intent
            ])
    
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_analytics_export.csv"}
    )