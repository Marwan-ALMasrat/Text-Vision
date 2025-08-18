import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

# إعداد الصفحة
st.set_page_config(
    page_title="Text Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل CSS مخصص
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #e9ecef !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# كلاس لتحميل النماذج
@st.cache_resource
def load_models():
    """تحميل النماذج المدربة"""
    try:
        # تحميل نموذج LSTM
        model = tf.keras.models.load_model('models/lstm_simple.h5')
        
        # تحميل Tokenizer
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # تحميل Label Encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # تحميل نموذج التلخيص
        summarizer = pipeline("summarization", 
                            model="facebook/bart-large-cnn",
                            device=-1)  # CPU
        
        return model, tokenizer, label_encoder, summarizer
    
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {e}")
        return None, None, None, None

# دوال معالجة النصوص
def preprocess_text(text, tokenizer, max_len=100):
    """معالجة النص للتصنيف"""
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded

def classify_text(text, model, tokenizer, label_encoder):
    """تصنيف النص"""
    try:
        processed_text = preprocess_text(text, tokenizer)
        prediction = model.predict(processed_text, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        class_name = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return class_name, confidence, prediction[0]
    
    except Exception as e:
        st.error(f"خطأ في التصنيف: {e}")
        return None, None, None

def summarize_text(text, summarizer, max_length=130, min_length=30):
    """تلخيص النص"""
    try:
        if len(text.split()) < 50:
            return "النص قصير جداً للتلخيص (يحتاج أكثر من 50 كلمة)"
        
        summary = summarizer(text, 
                           max_length=max_length, 
                           min_length=min_length, 
                           do_sample=False)
        
        return summary[0]['summary_text']
    
    except Exception as e:
        st.error(f"خطأ في التلخيص: {e}")
        return "حدث خطأ في عملية التلخيص"

def main():
    # العنوان الرئيسي
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Text Vision</h1>
        <p>منصة ذكية لتصنيف وتلخيص النصوص باستخدام الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # تحميل النماذج
    model, tokenizer, label_encoder, summarizer = load_models()
    
    if model is None:
        st.error("⚠️ فشل في تحميل النماذج. تأكد من وجود ملفات النماذج في المجلد المناسب.")
        st.stop()
    
    # الشريط الجانبي
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png")
        st.title("⚙️ الإعدادات")
        
        # اختيار العملية
        operation = st.selectbox(
            "اختر العملية:",
            ["🔍 تصنيف النص", "📝 تلخيص النص", "🔄 كلاهما"]
        )
        
        # إعدادات التلخيص
        if operation in ["📝 تلخيص النص", "🔄 كلاهما"]:
            st.subheader("إعدادات التلخيص")
            max_length = st.slider("الحد الأقصى لطول الملخص:", 50, 300, 130)
            min_length = st.slider("الحد الأدنى لطول الملخص:", 20, 100, 30)
        
        # معلومات إضافية
        with st.expander("ℹ️ معلومات"):
            st.info("""
            **Text Vision** يستخدم:
            - LSTM لتصنيف النصوص
            - BART لتلخيص النصوص
            - واجهة سهلة وجذابة
            """)
    
    # المحتوى الرئيسي
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 أدخل النص هنا:")
        
        # مربع النص
        text_input = st.text_area(
            "النص:",
            height=200,
            placeholder="اكتب أو الصق النص هنا...",
            help="أدخل النص الذي تريد تصنيفه أو تلخيصه"
        )
        
        # أمثلة سريعة
        st.subheader("🎯 أمثلة سريعة:")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("📰 مثال إخباري"):
                text_input = "أعلنت وزارة التعليم عن إطلاق برنامج جديد لتطوير المناهج الدراسية..."
        
        with col_ex2:
            if st.button("💼 مثال تجاري"):
                text_input = "شهدت الأسواق المالية اليوم تقلبات حادة بسبب التطورات الاقتصادية..."
        
        with col_ex3:
            if st.button("⚽ مثال رياضي"):
                text_input = "انتهت مباراة اليوم بفوز الفريق المحلي بهدفين مقابل هدف واحد..."
    
    with col2:
        st.subheader("📊 إحصائيات النص:")
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            
            # عرض الإحصائيات
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("عدد الكلمات", word_count)
            with col_stat2:
                st.metric("عدد الأحرف", char_count)
            
            # مؤشر طول النص
            if word_count < 50:
                st.warning("⚠️ النص قصير للتلخيص")
            elif word_count > 500:
                st.info("ℹ️ النص طويل - قد يستغرق وقتاً أطول")
            else:
                st.success("✅ طول النص مناسب")
    
    # زر المعالجة
    st.markdown("---")
    
    if st.button("🚀 ابدأ المعالجة", type="primary", use_container_width=True):
        if not text_input:
            st.warning("⚠️ يرجى إدخال النص أولاً")
            return
        
        # شريط التحميل
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # النتائج
        if operation in ["🔍 تصنيف النص", "🔄 كلاهما"]:
            status_text.text("جاري تصنيف النص...")
            progress_bar.progress(25)
            
            class_name, confidence, all_probs = classify_text(
                text_input, model, tokenizer, label_encoder
            )
            
            if class_name:
                st.markdown("### 🏷️ نتيجة التصنيف:")
                
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>الفئة المتوقعة: <span style="color: #667eea;">{class_name}</span></h4>
                        <h5>مستوى الثقة: {confidence:.2f}%</h5>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    # رسم بياني للتوزيع
                    class_names = label_encoder.classes_
                    fig = px.bar(
                        x=class_names, 
                        y=all_probs * 100,
                        title="توزيع الاحتمالات لجميع الفئات",
                        labels={'x': 'الفئات', 'y': 'الاحتمال (%)'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        progress_bar.progress(50)
        
        if operation in ["📝 تلخيص النص", "🔄 كلاهما"]:
            status_text.text("جاري تلخيص النص...")
            progress_bar.progress(75)
            
            summary = summarize_text(text_input, summarizer, max_length, min_length)
            
            st.markdown("### 📋 الملخص:")
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size: 16px; line-height: 1.6;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar.progress(100)
        status_text.text("✅ تم الانتهاء بنجاح!")
        
        # إزالة شريط التحميل بعد ثانيتين
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()