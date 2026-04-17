"""
Machine Learning Engine for Intent Classification
Phân loại ý định người dùng sử dụng Machine Learning
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from underthesea import word_tokenize


class IntentClassifier:
    """
    Bộ phân loại intent sử dụng Machine Learning
    """
    
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.intents: List[str] = []
        self.is_trained = False
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset từ file CSV
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        
        # Validate columns
        if 'text' not in df.columns or 'intent' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'intent' columns")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove NaN values
        df = df.dropna(subset=['text', 'intent'])
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản tiếng Việt
        """
        if not text:
            return ""
        
        # Tokenize tiếng Việt
        tokens = word_tokenize(text.lower())
        
        # Join lại thành chuỗi
        return " ".join(tokens)
    
    def train(self, dataset_path: str, test_size: float = 0.2) -> Dict:
        """
        Huấn luyện model phân loại intent
        
        Args:
            dataset_path: Đường dẫn đến file CSV dataset
            test_size: Tỷ lệ dữ liệu test
        
        Returns:
            Dictionary chứa metrics đánh giá model
        """
        print(f"📚 Loading dataset from {dataset_path}...")
        df = self.load_dataset(dataset_path)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"📊 Intents distribution:\n{df['intent'].value_counts()}")
        
        # Preprocess texts
        print("\n🔄 Preprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'],
            df['intent'],
            test_size=test_size,
            random_state=42,
            stratify=df['intent']
        )
        
        print(f"\n📈 Training set: {len(X_train)} samples")
        print(f"📉 Test set: {len(X_test)} samples")
        
        # Create pipeline: TF-IDF + Classifier
        print("\n🤖 Training model...")
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams và bigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Classifier (Logistic Regression thường tốt hơn cho text classification)
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Train
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\n✅ Training complete!")
        print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Lưu intents list
        self.intents = df['intent'].unique().tolist()
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'report': report,
            'num_samples': len(df),
            'num_intents': len(self.intents),
            'intents': self.intents
        }
    
    def predict(self, text: str) -> Dict:
        """
        Dự đoán intent của một câu
        
        Args:
            text: Câu cần phân loại
        
        Returns:
            Dictionary chứa intent dự đoán và confidence score
        """
        if not self.is_trained:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi train() trước.")
        
        # Preprocess
        processed = self.preprocess_text(text)
        
        # Vectorize
        vec = self.vectorizer.transform([processed])
        
        # Predict
        intent = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]
        
        # Get confidence score
        intent_idx = list(self.model.classes_).index(intent)
        confidence = probs[intent_idx]
        
        # Get top 3 intents
        top_indices = np.argsort(probs)[::-1][:3]
        top_intents = [
            {
                'intent': self.model.classes_[idx],
                'confidence': float(probs[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'intent': intent,
            'confidence': float(confidence),
            'top_intents': top_intents,
            'text': text,
            'processed_text': processed
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Dự đoán intent cho nhiều câu
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_dir: str = "models") -> str:
        """
        Lưu model và vectorizer ra disk
        """
        if not self.is_trained:
            raise ValueError("Không thể lưu model chưa được huấn luyện")
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "intent_classifier.joblib")
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        # Save model và vectorizer
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metadata
        import json
        metadata = {
            'intents': self.intents,
            'is_trained': self.is_trained,
            'num_intents': len(self.intents)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Model saved to {model_dir}/")
        return model_dir
    
    def load_model(self, model_dir: str = "models") -> bool:
        """
        Load model và vectorizer từ disk
        """
        model_path = os.path.join(model_dir, "intent_classifier.joblib")
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, metadata_path]):
            return False
        
        # Load model và vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Load metadata
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.intents = metadata['intents']
        self.is_trained = metadata['is_trained']
        
        print(f"✅ Model loaded from {model_dir}/")
        return True


# ==================== SINGLETON INSTANCE ====================
_intent_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """
    Get or create Intent Classifier instance (Singleton pattern)
    🔥 AUTO-LOAD MODEL khi khởi tạo nếu có file trong models/
    """
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
        
        # 🔥 Auto-load model nếu tồn tại
        if os.path.exists("models/intent_classifier.joblib"):
            try:
                _intent_classifier.load_model("models")
                print("✅ Intent Classifier: Model auto-loaded from models/")
            except Exception as e:
                print(f"⚠️ Intent Classifier: Could not auto-load model: {e}")
    
    return _intent_classifier


# ==================== TRAINING SCRIPT ====================
if __name__ == "__main__":
    # Đường dẫn đến dataset
    dataset_path = "data/intent_dataset.csv"
    model_dir = "models"
    
    print("="*60)
    print("🤖 MENTAL HEALTH CHATBOT - INTENT CLASSIFIER TRAINING")
    print("="*60)
    
    classifier = get_intent_classifier()
    
    # Kiểm tra xem đã có model chưa
    if os.path.exists(os.path.join(model_dir, "intent_classifier.joblib")):
        print("\n📂 Found existing model. Load or retrain?")
        choice = input("Enter 'l' to load, 'r' to retrain: ").strip().lower()
        
        if choice == 'l':
            if classifier.load_model(model_dir):
                print("✅ Model loaded successfully!")
                
                # Test với vài câu
                test_texts = [
                    "tôi buồn quá",
                    "tôi muốn chết",
                    "làm sao để giảm stress",
                    "chào bạn"
                ]
                
                print("\n🧪 Testing model:")
                for text in test_texts:
                    result = classifier.predict(text)
                    print(f"\nText: '{text}'")
                    print(f"Intent: {result['intent']} ({result['confidence']:.2%})")
                
                exit()
    
    # Train model mới
    print(f"\n📚 Training new model with dataset: {dataset_path}")
    metrics = classifier.train(dataset_path)
    
    # Lưu model
    print("\n💾 Saving model...")
    classifier.save_model(model_dir)
    
    # Test model
    print("\n" + "="*60)
    print("🧪 TESTING MODEL")
    print("="*60)
    
    test_texts = [
        "tôi buồn quá",
        "tôi cảm thấy chán nản",
        "tôi muốn tự tử",
        "tôi cần giúp đỡ",
        "làm sao để giảm stress",
        "chào bạn",
        "cảm ơn bạn",
        "tôi bị mất ngủ",
        "bạn là ai",
        "hôm nay tôi có chuyện này"
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\n💬 '{text}'")
        print(f"   → Intent: {result['intent']} ({result['confidence']:.2%})")
        if result['top_intents'][0]['confidence'] < 0.5:
            print(f"   ⚠️  Low confidence!")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)