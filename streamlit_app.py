import streamlit as st
import tensorflow as tf
import pickle
import os
import numpy as np
from transformers import pipeline
import logging
import traceback

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="نظام تصنيف وتلخيص النصوص",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            missing_files = []
            for path, name in [(model_path, 'النموذج'), (tokenizer_path, 'Tokenizer'), (encoder_path, 'Label Encoder')]:
                if not os.path.exists(path):
                    missing_files.append(f"ملف {name}: {path}")
            
            if missing_files:
                logger.warning("ملفات مفقودة:")
                for file in missing_files:
                    logger.warning(f"  - {file}")
                return None, None, None
            
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
            return None, None, None
    
    @st.cache_resource
    def load_summarization_model(_self):
        """تحميل نموذج التلخيص"""
        try:
            logger.info("جاري تحميل نموذج التلخيص...")
            
            # استخدام نموذج أصغر وأسرع للإنتاج
            summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",  # نموذج أصغر وأسرع
                device=-1,  # استخدام CPU
                framework="pt"
            )
            
            logger.info("✅ تم تحميل نموذج التلخيص بنجاح")
            return summarizer
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل نموذج التلخيص: {e}")
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

def preprocess_text(text, tokenizer, max_length=100):
    """معالجة النص قبل التصنيف"""
    try:
        # تحويل النص إلى تسلسل
        sequence = tokenizer.texts_to_sequences([text])
        
        # إضافة padding
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=max_length, padding='post'
        )
        
        return padded
    except Exception as e:
        logger.error(f"خطأ في معالجة النص: {e}")
        return None

def classify_text(text, model, tokenizer, label_encoder):
    """تصنيف النص"""
    try:
        # معالجة النص
        processed_text = preprocess_text(text, tokenizer)
        if processed_text is None:
            return None, None
        
        # التنبؤ
        prediction = model.predict(processed_text, verbose=0)
        
        # الحصول على الفئة المتوقعة
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(prediction[0][predicted_class_idx])
        
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"خطأ في تصنيف النص: {e}")
        return None, None

def summarize_text(text, summarizer, max_length=150, min_length=50):
    """تلخيص النص"""
    try:
        if len(text.split()) < 10:
            return "النص قصير جداً للتلخيص"
        
        # تقسيم النص إذا كان طويلاً جداً
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # تلخيص النص
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return summary[0]['summary_text']
        
    except Exception as e:
        logger.error(f"خطأ في تلخيص النص: {e}")
        return f"خطأ في التلخيص: {str(e)}"

def main():
    """الدالة الرئيسية للتطبيق"""
    
    # العنوان الرئيسي
    st.title("🤖 نظام تصنيف وتلخيص النصوص")
    st.markdown("---")
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")
        
        # معلومات حالة النماذج
        st.subheader("📊 حالة النماذج")
        
        # تحميل النماذج
        with st.spinner("جاري تحميل النماذج..."):
            model_loader = ModelLoader()
            classification_loaded = model_loader.load_all_models()
            model, tokenizer, label_encoder, summarizer = model_loader.get_models()
        
        # عرض حالة النماذج
        if model_loader.is_classification_ready():
            st.success("✅ نموذج التصنيف: متاح")
        else:
            st.error("❌ نموذج التصنيف: غير متاح")
        
        if model_loader.is_summarization_ready():
            st.success("✅ نموذج التلخيص: متاح")
        else:
            st.warning("⚠️ نموذج التلخيص: غير متاح")
        
        st.markdown("---")
        
        # خيارات التطبيق
        st.subheader("🎛️ خيارات التطبيق")
        show_confidence = st.checkbox("عرض نسبة الثقة", value=True)
        auto_summarize = st.checkbox("تلخيص تلقائي للنصوص الطويلة", value=False)
    
    # الواجهة الرئيسية
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 إدخال النص")
        
        # خيارات الإدخال
        input_method = st.radio(
            "اختر طريقة إدخال النص:",
            ["كتابة مباشرة", "رفع ملف نصي"],
            horizontal=True
        )
        
        user_text = ""
        
        if input_method == "كتابة مباشرة":
            user_text = st.text_area(
                "أدخل النص للتصنيف والتلخيص:",
                height=200,
                placeholder="اكتب النص هنا..."
            )
        else:
            uploaded_file = st.file_uploader(
                "اختر ملف نصي",
                type=['txt'],
                help="يدعم ملفات .txt فقط"
            )
            
            if uploaded_file is not None:
                try:
                    user_text = str(uploaded_file.read(), "utf-8")
                    st.success(f"تم تحميل الملف بنجاح! عدد الأحرف: {len(user_text)}")
                except Exception as e:
                    st.error(f"خطأ في قراءة الملف: {e}")
    
    with col2:
        st.header("ℹ️ معلومات النص")
        
        if user_text:
            words_count = len(user_text.split())
            chars_count = len(user_text)
            
            st.metric("عدد الكلمات", words_count)
            st.metric("عدد الأحرف", chars_count)
            
            if words_count > 200:
                st.info("النص طويل - مناسب للتلخيص")
            elif words_count < 10:
                st.warning("النص قصير - قد لا يكون التلخيص مفيداً")
    
    # زر المعالجة
    if st.button("🚀 تحليل النص", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ يرجى إدخال نص للتحليل")
        else:
            # تقسيم النتائج إلى أعمدة
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.subheader("🏷️ نتيجة التصنيف")
                
                if model_loader.is_classification_ready():
                    with st.spinner("جاري تصنيف النص..."):
                        predicted_class, confidence = classify_text(
                            user_text, model, tokenizer, label_encoder
                        )
                    
                    if predicted_class is not None:
                        st.success(f"**الفئة المتوقعة:** {predicted_class}")
                        if show_confidence:
                            st.info(f"**نسبة الثقة:** {confidence:.2%}")
                            
                            # شريط التقدم لنسبة الثقة
                            st.progress(confidence)
                    else:
                        st.error("❌ فشل في تصنيف النص")
                else:
                    st.error("❌ نموذج التصنيف غير متاح")
            
            with results_col2:
                st.subheader("📄 ملخص النص")
                
                if model_loader.is_summarization_ready():
                    with st.spinner("جاري تلخيص النص..."):
                        summary = summarize_text(user_text, summarizer)
                    
                    if summary:
                        st.success("**الملخص:**")
                        st.write(summary)
                    else:
                        st.error("❌ فشل في تلخيص النص")
                else:
                    st.warning("⚠️ نموذج التلخيص غير متاح")
    
    # معلومات إضافية
    st.markdown("---")
    with st.expander("ℹ️ معلومات حول النظام"):
        st.markdown("""
        ### حول النظام
        
        هذا النظام يقوم بـ:
        
        - **تصنيف النصوص**: استخدام نموذج LSTM لتصنيف النصوص إلى فئات مختلفة
        - **تلخيص النصوص**: استخدام نموذج BART المُدرب مسبقاً لتلخيص النصوص الطويلة
        
        ### المتطلبات التقنية
        
        - نموذج التصنيف: `lstm_simple.h5`
        - ملف المعالج: `tokenizer.pkl`  
        - ملف ترميز الفئات: `label_encoder.pkl`
        - نموذج التلخيص: `distilbart-cnn-12-6`
        
        ### نصائح الاستخدام
        
        - للحصول على أفضل النتائج في التصنيف، استخدم نصوص واضحة ومفهومة
        - للتلخيص، النصوص الطويلة (أكثر من 50 كلمة) تعطي نتائج أفضل
        - يمكنك رفع ملفات نصية بصيغة .txt مباشرة
        """)

    # معلومات أساسية في حال عدم توفر النماذج
    if not model_loader.is_classification_ready() and not model_loader.is_summarization_ready():
        st.error("""
        ❌ **لم يتم تحميل أي من النماذج بنجاح**
        
        يرجى التأكد من وجود ملفات النماذج في مجلد `models/`:
        - `lstm_simple.h5` (نموذج التصنيف)
        - `tokenizer.pkl` (معالج النصوص)
        - `label_encoder.pkl` (ترميز الفئات)
        
        أو تحقق من اتصال الإنترنت لتحميل نموذج التلخيص.
        """)

# تشغيل التطبيق
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"خطأ في تشغيل التطبيق: {e}")
        st.error("تفاصيل الخطأ:")
        st.code(traceback.format_exc())
