import re
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """كلاس لمعالجة النصوص"""
    
    def __init__(self, max_len=100):
        self.max_len = max_len
    
    @staticmethod
    def clean_text(text):
        """تنظيف وتحضير النص"""
        if not text or not isinstance(text, str):
            return ""
        
        # إزالة الأسطر الفارغة والمسافات الزائدة
        text = re.sub(r'\s+', ' ', text.strip())
        
        # إزالة الرموز الخاصة المتكررة
        text = re.sub(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?~`]{3,}', ' ', text)
        
        # الحفاظ على علامات الترقيم المهمة
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF.,!?;:]', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def validate_text(text, min_length=10, max_length=10000):
        """التحقق من صحة النص"""
        if not text:
            return False, "يرجى إدخال نص للمعالجة"
        
        if len(text) < min_length:
            return False, f"النص قصير جداً (الحد الأدنى {min_length} حرف)"
        
        if len(text) > max_length:
            return False, f"النص طويل جداً (الحد الأقصى {max_length} حرف)"
        
        # التحقق من وجود كلمات فعلية
        words = text.split()
        if len(words) < 3:
            return False, "النص يحتاج لأكثر من 3 كلمات"
        
        return True, "النص صالح للمعالجة"
    
    def preprocess_for_classification(self, text, tokenizer):
        """معالجة النص للتصنيف"""
        try:
            # تنظيف النص
            cleaned_text = self.clean_text(text)
            
            # تحويل إلى sequences
            sequences = tokenizer.texts_to_sequences([cleaned_text])
            
            # padding
            padded = pad_sequences(
                sequences, 
                maxlen=self.max_len, 
                padding='post', 
                truncating='post'
            )
            
            return padded
            
        except Exception as e:
            logger.error(f"خطأ في معالجة النص للتصنيف: {e}")
            raise e
    
    @staticmethod
    def preprocess_for_summarization(text, max_chunk_size=1024):
        """معالجة النص للتلخيص"""
        try:
            # تنظيف النص
            cleaned_text = TextProcessor.clean_text(text)
            
            # تقسيم النص إذا كان طويلاً جداً
            if len(cleaned_text) > max_chunk_size:
                # تقسيم عند الجمل
                sentences = re.split(r'[.!?]', cleaned_text)
                
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < max_chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
            
            return [cleaned_text]
            
        except Exception as e:
            logger.error(f"خطأ في معالجة النص للتلخيص: {e}")
            return [text]  # إرجاع النص الأصلي في حالة الخطأ
    
    @staticmethod
    def get_text_statistics(text):
        """حساب إحصائيات النص"""
        if not text:
            return {
                'characters': 0,
                'words': 0,
                'sentences': 0,
                'paragraphs': 0,
                'avg_word_length': 0,
                'reading_time': 0
            }
        
        # عدد الأحرف
        characters = len(text)
        
        # عدد الكلمات
        words = len(text.split())
        
        # عدد الجمل
        sentences = len(re.findall(r'[.!?]+', text))
        
        # عدد الفقرات
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        
        # متوسط طول الكلمة
        word_lengths = [len(word) for word in text.split()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # وقت القراءة التقديري (200 كلمة في الدقيقة)
        reading_time = round(words / 200, 1) if words > 0 else 0
        
        return {
            'characters': characters,
            'words': words,
            'sentences': max(sentences, 1),  # على الأقل جملة واحدة
            'paragraphs': max(paragraphs, 1),  # على الأقل فقرة واحدة
            'avg_word_length': round(avg_word_length, 1),
            'reading_time': reading_time
        }
    
    @staticmethod
    def classify_text_complexity(word_count):
        """تصنيف مستوى تعقيد النص"""
        if word_count < 50:
            return "قصير", "🟢"
        elif word_count < 200:
            return "متوسط", "🟡"
        elif word_count < 500:
            return "طويل", "🟠"
        else:
            return "طويل جداً", "🔴"
    
    @staticmethod
    def suggest_processing_options(text_stats):
        """اقتراح خيارات المعالجة بناء على النص"""
        suggestions = []
        
        word_count = text_stats['words']
        
        # اقتراحات للتصنيف
        if word_count >= 10:
            suggestions.append("✅ مناسب للتصنيف")
        else:
            suggestions.append("⚠️ قصير للتصنيف (يحتاج 10+ كلمات)")
        
        # اقتراحات للتلخيص
        if word_count >= 100:
            suggestions.append("✅ مناسب للتلخيص")
        elif word_count >= 50:
            suggestions.append("🟡 قابل للتلخيص (قد يكون مختصراً)")
        else:
            suggestions.append("❌ قصير جداً للتلخيص (يحتاج 50+ كلمة)")
        
        # اقتراحات للسرعة
        if word_count > 1000:
            suggestions.append("⏰ قد يستغرق معالجة أطول")
        
        return suggestions