"""
Risk Detection Module for Mental Health Chatbot
Phát hiện và phân loại mức độ nguy cơ tâm lý
"""

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import re


class RiskLevel(Enum):
    """Các mức độ nguy cơ"""
    NONE = "none"           # Không có nguy cơ
    LOW = "low"             # Nguy cơ thấp - theo dõi
    MEDIUM = "medium"       # Nguy cơ trung bình - can thiệp nhẹ
    HIGH = "high"           # Nguy cơ cao - can thiệp tích cực
    CRITICAL = "critical"   # Khẩn cấp - cần hỗ trợ ngay


@dataclass
class RiskAssessment:
    """Kết quả đánh giá nguy cơ"""
    level: RiskLevel
    score: float  # 0.0 - 1.0
    indicators: List[str]  # Các dấu hiệu phát hiện
    recommended_actions: List[str]  # Hành động đề xuất
    requires_human_intervention: bool
    emergency_contacts: List[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['level'] = self.level.value
        return result


class RiskDetector:
    """
    Bộ phát hiện nguy cơ tâm lý dựa trên rules + keywords
    """
    
    def __init__(self):
        # Crisis keywords - từ ngữ chỉ tình huống khẩn cấp
        self.crisis_keywords = {
            'suicide': [
                'tự tử', 'tự vẫn', 'quyên sinh', 'muốn chết', 'chết đi',
                'không muốn sống', 'sống làm gì', 'vô nghĩa', 'kết thúc',
                'nhảy lầu', 'cắt cổ', 'uống thuốc', 'thắt cổ'
            ],
            'self_harm': [
                'tự hại', 'làm đau bản thân', 'cắt tay', 'đập đầu',
                'đốt mình', 'cào cấu', 'tự làm tổn thương', 'self-harm'
            ],
            'severe_depression': [
                'trầm cảm nặng', 'không thể dậy', 'không ăn không ngủ',
                'mất hết hy vọng', 'tuyệt vọng', 'vô dụng', 'gánh nặng',
                'ai cũng tốt hơn mình', 'không ai hiểu', 'cô đơn cùng cực'
            ],
            'panic_attack': [
                'panic attack', 'khó thở dữ dội', 'tim đập như muốn vỡ',
                'chết đến nơi', 'sắp ngất', 'mất kiểm soát', 'hoang tưởng'
            ]
        }
        
        # Warning keywords - dấu hiệu cảnh báo sớm
        self.warning_keywords = {
            'depression_signs': [
                'buồn', 'chán', 'mệt mỏi', 'chán nản', 'suy sụp',
                'khóc', 'đau khổ', 'tổn thương', 'thất vọng', 'cô đơn'
            ],
            'anxiety_signs': [
                'lo', 'lo lắng', 'bồn chồn', 'bất an', 'hoang mang',
                'sợ', 'căng thẳng', 'áp lực', 'stress', 'run', 'đổ mồ hôi'
            ],
            'isolation_signs': [
                'không muốn gặp ai', 'tránh mặt', 'ở một mình',
                'không muốn nói chuyện', 'khép kín', 'cô lập'
            ],
            'sleep_appetite_issues': [
                'mất ngủ', 'khó ngủ', 'ngủ nhiều', 'ăn không ngon',
                'chán ăn', 'ăn quá nhiều', 'rối loạn giấc ngủ'
            ]
        }
        
        # Protective factors - yếu tố bảo vệ (giảm risk score)
        self.protective_keywords = [
            'cảm ơn', 'tốt hơn', 'hy vọng', 'cố gắng', 'sẽ ổn',
            'có người hỗ trợ', 'đi trị liệu', 'gặp bác sĩ', 'tự chăm sóc',
            'tập thể dục', 'thiền', 'viết nhật ký', 'nói chuyện với bạn'
        ]
        
        # Hành động đề xuất theo mức độ nguy cơ
        self.action_recommendations = {
            RiskLevel.NONE: [
                "Tiếp tục theo dõi cảm xúc",
                "Duy trì thói quen lành mạnh",
                "Chia sẻ với người thân khi cần"
            ],
            RiskLevel.LOW: [
                "Thực hành kỹ thuật thư giãn (thở sâu, mindfulness)",
                "Viết nhật ký cảm xúc",
                "Liên hệ bạn bè/người thân để chia sẻ",
                "Theo dõi triệu chứng trong 24-48h tới"
            ],
            RiskLevel.MEDIUM: [
                "⚠️ Cân nhắc gặp chuyên gia tâm lý",
                "Gọi đường dây nóng tư vấn: 111 (trẻ em), 1900 6194 (tâm lý)",
                "Tránh ở một mình quá lâu",
                "Thực hiện kế hoạch an toàn cá nhân"
            ],
            RiskLevel.HIGH: [
                "🆘 Cần hỗ trợ chuyên nghiệp NGAY",
                "Liên hệ người thân đáng tin cậy ngay lập tức",
                "Gọi 115 (cấp cứu) hoặc đến bệnh viện gần nhất",
                "Không ở một mình - nhờ ai đó ở bên cạnh"
            ],
            RiskLevel.CRITICAL: [
                "🚨 KHẨN CẤP - Gọi 115 NGAY LẬP TỨC",
                "Đến bệnh viện/cơ sở y tế gần nhất",
                "Liên hệ người thân/bạn bè để được hỗ trợ ngay",
                "Tổng đài quốc gia BVTE: 111 (24/7)",
                "Đường dây nóng phòng chống tự tử: 1900 6194"
            ]
        }
        
        # Emergency contacts Vietnam
        self.emergency_contacts = [
            "Cấp cứu y tế: 115",
            "Tổng đài quốc gia Bảo vệ Trẻ em: 111 (24/7)",
            "Đường dây nóng tư vấn tâm lý: 1900 6194",
            "Bệnh viện Tâm thần Trung ương: 024 3869 3795",
            "Viện Sức khỏe Tâm thần - Bạch Mai: 024 3576 5344"
        ]
    
    def detect_risk(self, text: str, context: Optional[Dict] = None) -> RiskAssessment:
        """
        Phân tích và đánh giá mức độ nguy cơ từ tin nhắn
        
        Args:
            text: Tin nhắn của người dùng
            context: Thông tin ngữ cảnh (tùy chọn)
        
        Returns:
            RiskAssessment object chứa kết quả đánh giá
        """
        if not text:
            return self._create_assessment(RiskLevel.NONE, 0.0, [])
        
        text_lower = text.lower()
        indicators = []
        risk_score = 0.0
        
        # 1. Check crisis keywords (trọng số cao)
        for category, keywords in self.crisis_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                indicators.extend([f"CRISIS[{category}]: {m}" for m in matches])
                risk_score += 0.4 * len(matches)  # Mỗi crisis keyword +0.4
        
        # 2. Check warning keywords (trọng số trung bình)
        for category, keywords in self.warning_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                indicators.extend([f"WARNING[{category}]: {m}" for m in matches])
                risk_score += 0.15 * len(matches)  # Mỗi warning keyword +0.15
        
        # 3. Check protective factors (giảm risk score)
        protective_matches = [kw for kw in self.protective_keywords if kw in text_lower]
        if protective_matches:
            indicators.extend([f"PROTECTIVE: {m}" for m in protective_matches])
            risk_score -= 0.1 * len(protective_matches)  # Giảm 0.1 cho mỗi protective factor
        
        # 4. Check intensity markers (từ nhấn mạnh)
        intensity_markers = ['rất', 'quá', 'lắm', 'vô cùng', 'cực kỳ', 'hoàn toàn']
        intensity_count = sum(1 for marker in intensity_markers if marker in text_lower)
        if intensity_count > 0 and risk_score > 0:
            risk_score *= (1 + 0.1 * intensity_count)  # Tăng 10% cho mỗi intensity marker
        
        # 5. Check context factors (nếu có)
        if context:
            # Previous high risk
            if context.get('previous_risk_level') in ['high', 'critical']:
                risk_score *= 1.2
                indicators.append("CONTEXT: Previous high risk detected")
            
            # Conversation length (càng dài càng cần chú ý)
            if context.get('message_count', 0) > 20:
                risk_score *= 1.1
                indicators.append("CONTEXT: Long conversation - monitor closely")
        
        # Clamp score to [0.0, 1.0]
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine risk level
        if risk_score >= 0.7:
            level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            level = RiskLevel.MEDIUM
        elif risk_score >= 0.1:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NONE
        
        # Create assessment
        return self._create_assessment(level, risk_score, indicators)
    
    def _create_assessment(self, level: RiskLevel, score: float, 
                          indicators: List[str]) -> RiskAssessment:
        """Helper: Tạo RiskAssessment object"""
        return RiskAssessment(
            level=level,
            score=round(score, 3),
            indicators=indicators[:10],  # Limit to top 10 indicators
            recommended_actions=self.action_recommendations[level],
            requires_human_intervention=level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            emergency_contacts=self.emergency_contacts if level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else None
        )
    
    def should_alert_human(self, assessment: RiskAssessment) -> bool:
        """Kiểm tra có cần cảnh báo nhân viên hỗ trợ không"""
        return assessment.requires_human_intervention
    
    def get_crisis_response(self, assessment: RiskAssessment) -> str:
        """Tạo phản hồi khẩn cấp khi phát hiện crisis"""
        if assessment.level not in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return ""
        
        base_response = "🚨 Tôi rất lo lắng cho sự an toàn của bạn. "
        
        if assessment.level == RiskLevel.CRITICAL:
            base_response += "Đây là tình huống KHẨN CẤP. "
        
        base_response += "\n\n" + " • ".join(assessment.recommended_actions[:3])
        
        if assessment.emergency_contacts:
            base_response += "\n\n📞 Số điện thoại khẩn cấp:\n"
            base_response += "\n".join(f"  - {contact}" for contact in assessment.emergency_contacts[:4])
        
        return base_response


# Singleton instance
_risk_detector = None

def get_risk_detector() -> RiskDetector:
    """Get or create Risk Detector instance (Singleton pattern)"""
    global _risk_detector
    if _risk_detector is None:
        _risk_detector = RiskDetector()
    return _risk_detector


# Test function
if __name__ == "__main__":
    detector = get_risk_detector()
    
    test_cases = [
        ("tôi buồn quá", "Normal sadness"),
        ("tôi muốn chết đi cho rồi", "Suicidal ideation"),
        ("tôi cắt tay để giảm đau", "Self-harm"),
        ("tôi lo lắng về kỳ thi", "Anxiety"),
        ("cuộc sống thật vô nghĩa, tôi không muốn sống nữa", "Severe depression"),
        ("cảm ơn bạn, tôi cảm thấy tốt hơn rồi", "Positive with protective factor"),
        ("tim tôi đập nhanh, khó thở, tôi sợ mình sắp chết", "Panic attack"),
    ]
    
    print("="*70)
    print("🔍 RISK DETECTOR - TEST CASES")
    print("="*70)
    
    for text, description in test_cases:
        print(f"\n📝 Test: {description}")
        print(f"💬 Input: '{text}'")
        
        assessment = detector.detect_risk(text)
        print(f"📊 Level: {assessment.level.value.upper()} (score: {assessment.score})")
        print(f"🔍 Indicators: {assessment.indicators[:3]}")
        print(f"✅ Actions: {assessment.recommended_actions[0]}")
        
        if assessment.requires_human_intervention:
            print("⚠️  REQUIRES HUMAN INTERVENTION!")
            print(f"📞 Emergency: {assessment.emergency_contacts[0]}")