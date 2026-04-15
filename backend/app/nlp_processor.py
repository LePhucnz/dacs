"""
Vietnamese NLP Processor for Mental Health Chatbot
Xử lý ngôn ngữ tự nhiên tiếng Việt
"""

import re
from typing import List, Dict
from underthesea import word_tokenize, sent_tokenize
from underthesea import pos_tag
import nltk
from nltk.corpus import stopwords

# Tải stopwords tiếng Việt nếu chưa có
try:
    stop_words_vi = set(nltk.corpus.stopwords.words('vietnamese'))
except:
    stop_words_vi = set()

class VietnameseNLP:
    """
    Bộ xử lý NLP tiếng Việt cho chatbot sức khỏe tâm thần
    """
    
    def __init__(self):
        self.stopwords = stop_words_vi
        
    def preprocess(self, text: str) -> str:
        """
        Tiền xử lý văn bản tiếng Việt
        - Lowercase
        - Remove special characters
        - Normalize whitespace
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\'\"\-\(\)]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize tiếng Việt sử dụng underthesea
        """
        if not text:
            return []
        
        processed = self.preprocess(text)
        tokens = word_tokenize(processed, format="text")
        
        return tokens.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Loại bỏ stopwords
        """
        # Vietnamese stopwords phổ biến
        vi_stopwords = {
            'là', 'mà', 'thì', 'và', 'hoặc', 'nhưng', 'với', 'của', 'cho',
            'trong', 'trên', 'dưới', 'từ', 'đến', 'tại', 'về', 'có', 'không',
            'phải', 'được', 'bị', 'đã', 'đang', 'sẽ', 'hãy', 'hết', 'rất',
            'quá', 'lắm', 'ra', 'vậy', 'thế', 'này', 'kia', 'ấy', 'đó',
            'tôi', 'tao', 'mày', 'nó', 'hắn', 'chúng', 'chúng ta', 'chúng tôi',
            'anh', 'chị', 'em', 'bạn', 'ông', 'bà', 'cô', 'chú', 'bác',
            'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười'
        }
        
        all_stopwords = self.stopwords.union(vi_stopwords)
        
        return [token for token in tokens if token not in all_stopwords and len(token) > 1]
    
    def pos_tagging(self, text: str) -> List[tuple]:
        """
        Part-of-Speech tagging cho tiếng Việt
        Returns: List of (word, POS_tag) tuples
        """
        if not text:
            return []
        
        tokens = self.tokenize(text)
        tagged = pos_tag(tokens)
        
        return tagged
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Trích xuất từ khóa quan trọng
        """
        tokens = self.tokenize(text)
        filtered = self.remove_stopwords(tokens)
        
        # Đếm tần suất
        from collections import Counter
        word_freq = Counter(filtered)
        
        # Lấy top_k từ phổ biến nhất
        keywords = [word for word, freq in word_freq.most_common(top_k)]
        
        return keywords
    
    def detect_emotion_words(self, text: str) -> Dict[str, List[str]]:
        """
        Phát hiện từ ngữ cảm xúc trong văn bản
        """
        # Danh sách từ cảm xúc tiếng Việt
        emotion_lexicon = {
            'negative': [
                'buồn', 'chán', 'stress', 'áp lực', 'lo lắng', 'sợ hãi',
                'trầm cảm', 'tuyệt vọng', 'cô đơn', 'mệt mỏi', 'đau khổ',
                'thất vọng', 'tự ti', 'lo âu', 'hoang mang', 'bất an',
                'khóc', 'đau', 'tổn thương', 'suy sụp', 'chán nản',
                'muốn chết', 'tự tử', 'không muốn sống', 'vô dụng'
            ],
            'positive': [
                'vui', 'hạnh phúc', 'yêu đời', 'lạc quan', 'tự tin',
                'hào hứng', 'thỏa mãn', 'bình yên', 'nhẹ nhõm', 'hy vọng',
                'cảm ơn', 'tốt', 'ổn', 'khỏe', 'cười', 'yêu'
            ],
            'anxiety': [
                'lo', 'lo lắng', 'bồn chồn', 'bất an', 'hoang mang',
                'sợ', 'sợ hãi', 'run', 'tim đập', 'khó thở',
                'căng thẳng', 'áp lực', 'stress', 'panic'
            ],
            'depression': [
                'buồn', 'chán', 'trầm cảm', 'tuyệt vọng', 'vô vọng',
                'cô đơn', 'cô lập', 'mệt mỏi', 'chán nản', 'suy sụp',
                'không muốn sống', 'muốn chết', 'tự tử', 'vô dụng',
                'tội lỗi', 'xấu hổ', 'thất vọng'
            ]
        }
        
        tokens = self.tokenize(text.lower())
        detected = {
            'negative': [],
            'positive': [],
            'anxiety': [],
            'depression': []
        }
        
        for emotion_type, words in emotion_lexicon.items():
            for word in words:
                if word in tokens or word in text.lower():
                    detected[emotion_type].append(word)
        
        return detected
    
    def calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Tính điểm sentiment (cảm xúc) của văn bản
        Returns: Dictionary với scores cho các loại cảm xúc
        """
        emotions = self.detect_emotion_words(text)
        
        total_words = len(self.tokenize(text))
        if total_words == 0:
            return {
                'negative_score': 0.0,
                'positive_score': 0.0,
                'anxiety_score': 0.0,
                'depression_score': 0.0,
                'overall_sentiment': 'neutral'
            }
        
        negative_score = len(emotions['negative']) / total_words
        positive_score = len(emotions['positive']) / total_words
        anxiety_score = len(emotions['anxiety']) / total_words
        depression_score = len(emotions['depression']) / total_words
        
        # Xác định sentiment tổng thể
        if depression_score > 0.1 or negative_score > 0.15:
            overall = 'very_negative'
        elif negative_score > 0.05:
            overall = 'negative'
        elif positive_score > 0.05:
            overall = 'positive'
        else:
            overall = 'neutral'
        
        return {
            'negative_score': round(negative_score, 3),
            'positive_score': round(positive_score, 3),
            'anxiety_score': round(anxiety_score, 3),
            'depression_score': round(depression_score, 3),
            'overall_sentiment': overall,
            'detected_emotions': emotions
        }


# Singleton instance
_nlp_processor = None

def get_nlp_processor() -> VietnameseNLP:
    """
    Get or create NLP processor instance (Singleton pattern)
    """
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = VietnameseNLP()
    return _nlp_processor


# Test function
if __name__ == "__main__":
    nlp = get_nlp_processor()
    
    # Test cases
    test_texts = [
        "Tôi cảm thấy rất buồn và chán nản. Tôi không biết phải làm sao nữa.",
        "Hôm nay tôi vui quá! Mọi thứ thật tuyệt vời.",
        "Tôi lo lắng về kỳ thi sắp tới. Tim tôi đập nhanh và khó thở.",
        "Cuộc sống thật vô nghĩa. Tôi muốn chết đi cho rồi."
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        print(f"Tokens: {nlp.tokenize(text)[:10]}...")
        print(f"Keywords: {nlp.extract_keywords(text)}")
        
        sentiment = nlp.calculate_sentiment_score(text)
        print(f"Sentiment: {sentiment['overall_sentiment']}")
        print(f"Negative: {sentiment['negative_score']}, Positive: {sentiment['positive_score']}")
        
        if sentiment['detected_emotions']['depression']:
            print(f"⚠️ Depression words: {sentiment['detected_emotions']['depression']}")
        if sentiment['detected_emotions']['anxiety']:
            print(f"⚠️ Anxiety words: {sentiment['detected_emotions']['anxiety']}")