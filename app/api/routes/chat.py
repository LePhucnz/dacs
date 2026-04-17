"""
Chat endpoints for Mental Health Chatbot
Endpoint /chat/friend - Trò chuyện với chatbot đồng cảm + Context Awareness + AUTH
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import CurrentUser  # 🔐 Import dependency xác thực
from sqlmodel import Session
from sqlalchemy import create_engine
from app.core.config import settings
from app.crud.message import create_message, get_messages_by_session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import uuid

# Database engine và session
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

def get_db():
    """Dependency để lấy DB session"""
    with Session(engine) as session:
        yield session

from app.nlp_processor import get_nlp_processor, VietnameseNLP
from app.empathy_engine import get_empathy_engine, EmpathyEngine
from app.ml_engine import get_intent_classifier, IntentClassifier
from app.risk_detector import get_risk_detector, RiskDetector, RiskLevel, RiskAssessment

router = APIRouter(prefix="/chat", tags=["chat"])


# ==================== PYDANTIC MODELS ====================

class ChatMessage(BaseModel):
    """Tin nhắn chat cơ bản"""
    role: str = Field(..., description="role: 'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatRequest(BaseModel):
    """Request body cho endpoint chat"""
    message: str = Field(..., min_length=1, max_length=2000, 
                        description="Tin nhắn của người dùng")
    conversation_id: Optional[str] = Field(None, description="ID cuộc hội thoại")
    user_id: Optional[str] = Field(None, description="ID người dùng (nếu đã login)")
    context: Optional[Dict] = Field(None, description="Thông tin ngữ cảnh bổ sung")


class ChatResponse(BaseModel):
    """Response body cho endpoint chat"""
    response: str = Field(..., description="Phản hồi từ chatbot")
    conversation_id: str = Field(..., description="ID cuộc hội thoại")
    message_id: str = Field(..., description="ID tin nhắn phản hồi")
    
    # Metadata phân tích
    intent: Optional[str] = Field(None, description="Intent được phân loại")
    intent_confidence: Optional[float] = Field(None, description="Độ tin cậy của intent")
    
    sentiment: Optional[str] = Field(None, description="Sentiment tổng thể")
    sentiment_scores: Optional[Dict[str, float]] = Field(None, description="Chi tiết sentiment scores")
    
    risk_assessment: Optional[Dict] = Field(None, description="Đánh giá nguy cơ tâm lý")
    
    suggested_actions: Optional[List[str]] = Field(None, description="Hành động đề xuất")
    
    # 🔥 Context Window fields
    context_used: bool = Field(default=False, description="Có sử dụng lịch sử để phân tích không")
    context_snippet: Optional[str] = Field(None, description="Đoạn context đã dùng (cắt ngắn)")
    
    # Flags quan trọng
    is_crisis: bool = Field(default=False, description="Có phát hiện tình huống khẩn cấp không")
    requires_human_intervention: bool = Field(default=False, description="Cần can thiệp của con người không")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    """Lưu trữ lịch sử hội thoại"""
    conversation_id: str
    user_id: Optional[str] = None
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)


# ==================== IN-MEMORY STORAGE (Dev only) ====================

_conversations: Dict[str, ConversationHistory] = {}


def get_conversation(conversation_id: str) -> Optional[ConversationHistory]:
    """Lấy conversation từ memory"""
    return _conversations.get(conversation_id)


def save_conversation(conversation: ConversationHistory) -> None:
    """Lưu conversation vào memory"""
    conversation.updated_at = datetime.utcnow()
    _conversations[conversation.conversation_id] = conversation


def create_new_conversation(user_id: Optional[str] = None) -> ConversationHistory:
    """Tạo conversation mới"""
    conv_id = str(uuid.uuid4())
    return ConversationHistory(
        conversation_id=conv_id,
        user_id=user_id,
        messages=[]
    )


# ==================== CHATBOT CORE LOGIC ====================

class ChatbotCore:
    """Core logic xử lý chat - tích hợp tất cả modules + Context Window"""
    
    def __init__(self):
        self.nlp = get_nlp_processor()
        self.empathy = get_empathy_engine()
        self.intent_classifier = get_intent_classifier()
        self.risk_detector = get_risk_detector()
    
    def _build_context_input(self, conversation: Optional[ConversationHistory], 
                            current_message: str, 
                            context_window: int = 3) -> tuple[str, bool, Optional[str]]:
        """🔥 XÂY DỰNG INPUT CÓ CONTEXT CHO ML/NLP"""
        if not conversation or len(conversation.messages) < 2:
            return current_message, False, None
        
        recent_messages = conversation.messages[-context_window:]
        context_parts = []
        for msg in recent_messages:
            role = "bạn" if msg.role == "user" else "tôi"
            context_parts.append(f"{role}: {msg.content}")
        
        context_text = " | ".join(context_parts)
        input_with_context = f"{context_text} | bạn: {current_message}"
        snippet = context_text[-200:] if len(context_text) > 200 else context_text
        
        return input_with_context, True, snippet
    
    def process_message(self, user_message: str, 
                       conversation: Optional[ConversationHistory] = None,
                       context: Optional[Dict] = None) -> ChatResponse:
        """Xử lý tin nhắn và tạo phản hồi thông minh CÓ CONTEXT"""
        input_for_analysis, context_used, context_snippet = self._build_context_input(
            conversation, user_message
        )
        
        # 1. NLP Analysis
        sentiment_analysis = self.nlp.calculate_sentiment_score(input_for_analysis)
        
        # 2. Intent Classification
        intent_result = None
        if self.intent_classifier.is_trained:
            try:
                intent_result = self.intent_classifier.predict(input_for_analysis)
            except Exception:
                intent_result = {'intent': 'unknown', 'confidence': 0.0}
        
        # 3. Risk Assessment
        risk_context = {}
        if conversation:
            risk_context['previous_risk_level'] = conversation.metadata.get('last_risk_level')
            risk_context['message_count'] = len(conversation.messages)
            risk_context['context_used'] = context_used
        
        risk_assessment = self.risk_detector.detect_risk(user_message, risk_context)
        
        # 4. Generate Empathetic Response
        empathy_response = self.empathy.generate_empathetic_response(
            user_message, sentiment_analysis
        )
        
        # 5. Handle Crisis Response
        final_response = empathy_response['response']
        if risk_assessment.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            crisis_msg = self.risk_detector.get_crisis_response(risk_assessment)
            if crisis_msg:
                final_response = crisis_msg
        
        conversation_id = conversation.conversation_id if conversation else str(uuid.uuid4())
        
        return ChatResponse(
            response=final_response,
            conversation_id=conversation_id,
            message_id=str(uuid.uuid4()),
            intent=intent_result['intent'] if intent_result else None,
            intent_confidence=intent_result['confidence'] if intent_result else None,
            sentiment=sentiment_analysis['overall_sentiment'],
            sentiment_scores={
                'negative': sentiment_analysis['negative_score'],
                'positive': sentiment_analysis['positive_score'],
                'anxiety': sentiment_analysis['anxiety_score'],
                'depression': sentiment_analysis['depression_score']
            },
            risk_assessment=risk_assessment.to_dict(),
            suggested_actions=empathy_response.get('suggested_actions', []),
            context_used=context_used,
            context_snippet=context_snippet,
            is_crisis=risk_assessment.level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            requires_human_intervention=risk_assessment.requires_human_intervention
        )


# Singleton chatbot instance
_chatbot_core = None

def get_chatbot_core() -> ChatbotCore:
    """Get or create Chatbot Core instance"""
    global _chatbot_core
    if _chatbot_core is None:
        _chatbot_core = ChatbotCore()
    return _chatbot_core


# ==================== API ENDPOINTS ====================

@router.post("/friend", response_model=ChatResponse, 
             summary="Chat with Mental Health Friend",
             description="Gửi tin nhắn và nhận phản hồi đồng cảm từ chatbot sức khỏe tâm thần")
async def chat_with_friend(
    request: ChatRequest,
    *,  # ✅ FIX: Thêm *, để tách positional và keyword-only params
    chatbot: ChatbotCore = Depends(get_chatbot_core),
    db: Session = Depends(get_db),
    current_user: CurrentUser,  # ✅ OK: current_user giờ là keyword-only
):
    """Endpoint chính cho chatbot (đã yêu cầu xác thực)"""
    
    # 💡 Mẹo: Bạn có thể dùng current_user.id thay vì request.user_id
    active_user_id = str(current_user.id)
    
    # Load hoặc tạo conversation
    conversation = None
    if request.conversation_id and request.conversation_id in _conversations:
        conversation = get_conversation(request.conversation_id)
    else:
        conversation = create_new_conversation(active_user_id)
        if request.conversation_id:
            conversation.conversation_id = request.conversation_id
    
    # Thêm user message vào conversation
    user_msg = ChatMessage(role="user", content=request.message)
    conversation.messages.append(user_msg)
    
    # Process message (có context window)
    response = chatbot.process_message(
        user_message=request.message,
        conversation=conversation,
        context=request.context
    )
    
    # Thêm bot response vào conversation
    bot_msg = ChatMessage(role="assistant", content=response.response)
    conversation.messages.append(bot_msg)
    
    # Update conversation metadata
    conversation.metadata.update({
        'last_risk_level': response.risk_assessment['level'],
        'last_sentiment': response.sentiment,
        'last_intent': response.intent,
        'crisis_count': conversation.metadata.get('crisis_count', 0) + (1 if response.is_crisis else 0)
    })
    
    # Save conversation (in-memory)
    save_conversation(conversation)
    
    # Alert nếu cần human intervention
    if response.requires_human_intervention:
        pass  # Placeholder for alert logic
    
    # 💾 LƯU TIN NHẮN VÀO POSTGRESQL
    create_message(
        db=db,
        session_id=response.conversation_id,
        user_message=request.message,
        bot_response=response.response,
        intent=response.intent,
        intent_confidence=response.intent_confidence,
        sentiment=response.sentiment,
        sentiment_scores=response.sentiment_scores,
        risk_level=response.risk_assessment["level"],
        risk_score=response.risk_assessment["score"],
        is_crisis=response.is_crisis,
        requires_human_intervention=response.requires_human_intervention,
        context_used=response.context_used,
        context_snippet=response.context_snippet,
        user_id=active_user_id,  # ✅ Dùng ID từ user đã đăng nhập
    )
    
    return response


@router.get("/friend/{conversation_id}", response_model=ConversationHistory,
            summary="Get Conversation History")
async def get_conversation_history(
    conversation_id: str,
):
    """Lấy lịch sử hội thoại theo ID"""
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/friend/{conversation_id}", summary="Delete Conversation")
async def delete_conversation(
    conversation_id: str,
):
    """Xóa hội thoại"""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del _conversations[conversation_id]
    return {"message": "Conversation deleted", "conversation_id": conversation_id}


@router.get("/health", summary="Chat Service Health Check")
async def chat_health_check():
    """Kiểm tra trạng thái chat service"""
    chatbot = get_chatbot_core()
    return {
        "status": "ok",
        "modules": {
            "nlp_processor": True,
            "empathy_engine": True,
            "intent_classifier": chatbot.intent_classifier.is_trained,
            "risk_detector": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }