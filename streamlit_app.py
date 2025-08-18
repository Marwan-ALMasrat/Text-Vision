import streamlit as st
import tensorflow as tf
import pickle
import os
from transformers import pipeline
import logging

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """كلاس لتحميل وإدارة النماذج"""
    
    def __init__(self, models_path='models/'):
        self.models_path = models_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.summarizer = None
        
    @st.cache_resource
    def load_classification_model(_self):
        """تحميل نموذج التصنيف و Tokenizer و Label Encoder"""
        try:
            model_path = os.path.join(_self.models_path, 'lstm_simple.h5')
            tokenizer_path = os.path.join(_self.models_path, 'tokenizer.pkl')
            encoder_path = os.path.join(_self.models_path, 'label_encoder.pkl')
            
            # التحقق من وجود الملفات
            for path, name in [(model_path, 'النموذج'), (tokenizer_path, 'Tokenizer'), (encoder_path, 'Label Encoder')]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"ملف {name} غير موجود: {path}")
            
            # تحميل النموذج
            logger.info("جاري تحميل نموذج التصنيف...")
            model = tf.keras.models.load_model(model_path)
            
            # تحميل Tokenizer
            logger.info("جاري تحميل Tokenizer...")
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # تحميل Label Encoder
            logger.info("جاري تحميل Label Encoder...")
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            logger.info("✅ تم تحميل نماذج التصنيف بنجاح")
            return model, tokenizer, label_encoder
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نماذج التصنيف: {e}")
            st.error(f"خطأ في تحميل نماذج التصنيف: {e}")
            return None, None, None
    
    @st.cache_resource
    def load_summarization_model(_self):
        """تحميل نموذج التلخيص"""
        try:
            logger.info("جاري تحميل نموذج التلخيص...")
            
            # استخدام نموذج أصغر وأسرع للإنتاج
            summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1,  # استخدام CPU
                framework="pt"
            )
            
            logger.info("✅ تم تحميل نموذج التلخيص بنجاح")
            return summarizer
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج التلخيص: {e}")
            
            # محاولة استخدام نموذج بديل أصغر
            try:
                logger.info("محاولة تحميل نموذج بديل...")
                summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1
                )
                st.warning("تم استخدام نموذج تلخيص بديل (أصغر)")
                return summarizer
            except:
                st.error("فشل في تحميل نموذج التلخيص. سيتم تعطيل ميزة التلخيص.")
                return None
    
    def load_all_models(self):
        """تحميل جميع النماذج"""
        # تحميل نماذج التصنيف
        self.model, self.tokenizer, self.label_encoder = self.load_classification_model()
        
        # تحميل نموذج التلخيص
        self.summarizer = self.load_summarization_model()
        
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.label_encoder is not None)
    
    def get_models(self):
        """إرجاع جميع النماذج"""
        return self.model, self.tokenizer, self.label_encoder, self.summarizer
    
    def is_classification_ready(self):
        """التحقق من جاهزية نماذج التصنيف"""
        return all([self.model, self.tokenizer, self.label_encoder])
    
    def is_summarization_ready(self):
        """التحقق من جاهزية نموذج التلخيص"""
        return self.summarizer is not None
