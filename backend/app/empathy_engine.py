"""
Empathy Engine for Mental Health Chatbot
Hệ thống phản hồi đồng cảm dựa trên rules
"""

from typing import Dict, List, Optional
import random
from app.nlp_processor import get_nlp_processor


class EmpathyEngine:
    """
    Engine tạo phản hồi đồng cảm cho chatbot sức khỏe tâm thần
    Sử dụng rule-based approach kết hợp với NLP
    """
    
    def __init__(self):
        self.nlp = get_nlp_processor()
        
        # Templates cho các loại phản hồi đồng cảm
        self.empathy_templates = {
            'acknowledge_feeling': [
                "Tôi hiểu rằng bạn đang cảm thấy {feeling}. Điều đó thật sự khó khăn.",
                "Nghe có vẻ như bạn đang trải qua {feeling}. Tôi ở đây để lắng nghe bạn.",
                "Cảm giác {feeling} là hoàn toàn bình thường trong tình huống này.",
                "Tôi có thể cảm nhận được {feeling} của bạn. Bạn không cô đơn đâu."
            ],
            'validate_emotion': [
                "Cảm xúc của bạn là hợp lý và đáng được tôn trọng.",
                "Không có gì sai khi cảm thấy như vậy. Bạn đang làm rất tốt.",
                "Mọi người đều có những lúc như thế này. Bạn không một mình.",
                "Điều bạn đang trải qua là có thật và quan trọng."
            ],
            'encourage_expression': [
                "Bạn có thể chia sẻ thêm về điều này không? Tôi luôn lắng nghe.",
                "Nếu bạn muốn nói thêm, tôi sẵn sàng lắng nghe bạn.",
                "Hãy cứ chia sẻ những gì bạn cảm thấy. Đây là không gian an toàn.",
                "Bạn muốn kể thêm về điều gì khiến bạn cảm thấy như vậy không?"
            ],
            'offer_support': [
                "Tôi ở đây để hỗ trợ bạn. Bạn không cần phải đối mặt một mình.",
                "Chúng ta sẽ cùng nhau tìm cách vượt qua điều này.",
                "Bạn có thể tin tưởng tôi. Tôi luôn sẵn lòng giúp đỡ.",
                "Hãy nhớ rằng bạn xứng đáng được quan tâm và giúp đỡ."
            ],
            'calming_techniques': [
                "Hãy thử hít thở sâu: hít vào 4 giây, giữ 4 giây, thở ra 6 giây.",
                "Bạn có thể thử tập trung vào hiện tại: nhìn xung quanh và gọi tên 5 thứ bạn thấy.",
                "Thử thả lỏng cơ thể từ từ, bắt đầu từ ngón chân lên đến đầu.",
                "Uống một ngụm nước và hít thở chậm rãi có thể giúp bạn bình tĩnh hơn."
            ],
            'positive_affirmation': [
                "Bạn mạnh mẽ hơn bạn nghĩ rất nhiều.",
                "Mỗi ngày là một cơ hội mới. Bạn đã làm rất tốt khi vượt qua hôm qua.",
                "Cảm xúc của bạn là tạm thời, nhưng bạn thì vĩnh cửu và đáng giá.",
                "Bạn xứng đáng có được sự bình yên và hạnh phúc."
            ],
            'crisis_intervention': [
                "⚠️ Tôi rất lo lắng cho bạn. Hãy gọi ngay cho người thân hoặc bạn bè.",
                " Bạn cần sự giúp đỡ chuyên nghiệp ngay lập tức. Hãy gọi 111 (Tổng đài quốc gia BVTE).",
                "⚠️ Tính mạng là quan trọng nhất. Hãy đến bệnh viện gần nhất hoặc gọi 115.",
                "🆘 Đừng ở một mình lúc này. Hãy gọi cho ai đó bạn tin tưởng ngay bây giờ."
            ],
            'suggest_professional_help': [
                "Có lẽ bạn nên cân nhắc gặp chuyên gia tâm lý để được hỗ trợ tốt hơn.",
                "Các bác sĩ tâm lý có thể giúp bạn hiểu rõ hơn về cảm xúc của mình.",
                "Tìm kiếm sự giúp đỡ chuyên nghiệp là một bước đi dũng cảm.",
                "Bạn có thể liên hệ với các trung tâm tư vấn tâm lý uy tín."
            ]
        }
        
        # Mapping giữa emotion types và phản hồi
        self.emotion_response_map = {
            'very_negative': ['acknowledge_feeling', 'validate_emotion', 'offer_support'],
            'negative': ['acknowledge_feeling', 'encourage_expression', 'calming_techniques'],
            'anxiety': ['calming_techniques', 'positive_affirmation', 'offer_support'],
            'depression': ['validate_emotion', 'offer_support', 'suggest_professional_help'],
            'neutral': ['encourage_expression', 'offer_support'],
            'positive': ['positive_affirmation', 'encourage_expression']
        }
    
    def generate_empathetic_response(
        self, 
        user_message: str,
        sentiment_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Tạo phản hồi đồng cảm dựa trên tin nhắn của user
        
        Args:
            user_message: Tin nhắn từ người dùng
            sentiment_analysis: Kết quả phân tích sentiment (optional)
        
        Returns:
            Dictionary chứa response và metadata
        """
        # Phân tích sentiment nếu chưa có
        if sentiment_analysis is None:
            sentiment_analysis = self.nlp.calculate_sentiment_score(user_message)
        
        # Xác định loại phản hồi cần thiết
        sentiment = sentiment_analysis['overall_sentiment']
        detected_emotions = sentiment_analysis.get('detected_emotions', {})
        
        # Kiểm tra crisis (tự tử, tự hại)
        is_crisis = self._detect_crisis(user_message, detected_emotions)
        
        # Chọn templates phù hợp
        if is_crisis:
            response_templates = self.empathy_templates['crisis_intervention']
            priority = 'crisis'
        else:
            # Lấy templates dựa trên sentiment
            templates_list = self.emotion_response_map.get(
                sentiment, 
                ['acknowledge_feeling', 'encourage_expression']
            )
            
            # Chọn ngẫu nhiên 1-2 templates
            selected_templates = random.sample(
                templates_list, 
                min(2, len(templates_list))
            )
            
            # Generate responses
            responses = []
            for template_type in selected_templates:
                template_list = self.empathy_templates[template_type]
                response = random.choice(template_list)
                
                # Fill feeling nếu có
                if '{feeling}' in response:
                    feeling = self._extract_primary_feeling(detected_emotions, sentiment)
                    response = response.format(feeling=feeling)
                
                responses.append(response)
            
            response_templates = responses
            priority = 'normal'
        
        # Kết hợp responses
        final_response = " ".join(response_templates)
        
        return {
            'response': final_response,
            'sentiment': sentiment,
            'priority': priority,
            'is_crisis': is_crisis,
            'detected_emotions': detected_emotions,
            'suggested_actions': self._suggest_actions(sentiment, is_crisis)
        }
    
    def _detect_crisis(self, text: str, emotions: Dict) -> bool:
        """
        Phát hiện tình huống khủng hoảng (tự tử, tự hại)
        """
        crisis_keywords = [
            'muốn chết', 'tự tử', 'tự vẫn', 'quyên sinh',
            'không muốn sống', 'sống làm gì', 'chết đi',
            'tự hại', 'làm đau bản thân', 'cắt tay', 'uống thuốc'
        ]
        
        text_lower = text.lower()
        
        # Check keywords
        for keyword in crisis_keywords:
            if keyword in text_lower:
                return True
        
        # Check depression score cao
        if len(emotions.get('depression', [])) >= 3:
            return True
        
        return False
    
    def _extract_primary_feeling(self, emotions: Dict, sentiment: str) -> str:
        """
        Trích xuất cảm xúc chính từ danh sách detected emotions
        """
        if emotions.get('depression'):
            return 'buồn bã và tuyệt vọng'
        elif emotions.get('anxiety'):
            return 'lo lắng và bất an'
        elif emotions.get('negative'):
            return 'khó khăn'
        elif emotions.get('positive'):
            return 'vui vẻ'
        else:
            return 'như vậy'
    
    def _suggest_actions(self, sentiment: str, is_crisis: bool) -> List[str]:
        """
        Đề xuất hành động dựa trên sentiment
        """
        if is_crisis:
            return [
                "Gọi ngay cho người thân hoặc bạn bè",
                "Liên hệ tổng đài 111 (Tổng đài quốc gia Bảo vệ Trẻ em)",
                "Đến bệnh viện gần nhất",
                "Không ở một mình"
            ]
        
        actions_map = {
            'very_negative': [
                "Nghỉ ngơi và thư giãn",
                "Viết nhật ký về cảm xúc",
                "Gọi cho người bạn tin tưởng",
                "Cân nhắc gặp chuyên gia tâm lý"
            ],
            'negative': [
                "Tập thở sâu",
                "Đi dạo ngoài trời",
                "Nghe nhạc nhẹ nhàng",
                "Viết ra những điều bạn biết ơn"
            ],
            'anxiety': [
                "Tập trung vào hơi thở",
                "Thực hành mindfulness",
                "Uống trà ấm",
                "Làm điều gì đó bạn yêu thích"
            ],
            'depression': [
                "Vận động nhẹ nhàng",
                "Tiếp xúc với ánh sáng mặt trời",
                "Kết nối với bạn bè",
                "Tìm kiếm sự giúp đỡ chuyên nghiệp"
            ],
            'neutral': [
                "Tiếp tục chia sẻ nếu bạn muốn",
                "Làm điều gì đó khiến bạn vui",
                "Chăm sóc bản thân"
            ],
            'positive': [
                "Duy trì cảm xúc tích cực này",
                "Chia sẻ niềm vui với người khác",
                "Ghi lại những điều tốt đẹp"
            ]
        }
        
        return actions_map.get(sentiment, ["Chia sẻ thêm nếu bạn muốn"])
    
    def get_conversation_summary(self, messages: List[Dict]) -> Dict:
        """
        Tóm tắt cuộc hội thoại và xu hướng cảm xúc
        """
        if not messages:
            return {'summary': 'Chưa có tin nhắn', 'trend': 'unknown'}
        
        sentiments = []
        for msg in messages:
            if msg.get('role') == 'user':
                analysis = self.nlp.calculate_sentiment_score(msg.get('content', ''))
                sentiments.append(analysis['overall_sentiment'])
        
        # Xác định trend
        if not sentiments:
            trend = 'unknown'
        else:
            recent = sentiments[-3:]  # 3 tin nhắn gần nhất
            if recent.count('very_negative') > 0:
                trend = 'worsening'
            elif recent.count('positive') > len(recent) // 2:
                trend = 'improving'
            else:
                trend = 'stable'
        
        return {
            'total_messages': len(messages),
            'dominant_sentiment': max(set(sentiments), key=sentiments.count) if sentiments else 'unknown',
            'trend': trend,
            'crisis_detected': any(
                self._detect_crisis(msg.get('content', ''), {}) 
                for msg in messages 
                if msg.get('role') == 'user'
            )
        }


# Singleton instance
_empathy_engine = None

def get_empathy_engine() -> EmpathyEngine:
    """
    Get or create Empathy Engine instance (Singleton pattern)
    """
    global _empathy_engine
    if _empathy_engine is None:
        _empathy_engine = EmpathyEngine()
    return _empathy_engine


# Test function
if __name__ == "__main__":
    engine = get_empathy_engine()
    
    test_messages = [
        "Tôi cảm thấy rất buồn và không biết phải làm sao",
        "Hôm nay tôi lo lắng quá, tim đập nhanh và khó thở",
        "Tôi muốn chết đi cho rồi, cuộc sống thật vô nghĩa",
        "Cảm ơn bạn, tôi cảm thấy tốt hơn rồi"
    ]
    
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"User: {msg}")
        response = engine.generate_empathetic_response(msg)
        print(f"Bot: {response['response']}")
        print(f"Sentiment: {response['sentiment']}, Priority: {response['priority']}")
        if response['is_crisis']:
            print("⚠️ CRISIS DETECTED!")
        print(f"Suggested actions: {response['suggested_actions'][:2]}")