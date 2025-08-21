import streamlit as st
import time

# Force immediate display
st.set_page_config(
    page_title="Test App", 
    page_icon="🚀",
    layout="wide"
)

# Immediate content
st.write("🚀 **APP IS LOADING...**")

# Show timestamp to prove it's working
st.write(f"⏰ Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Test basic components
st.title("✅ Streamlit is Working!")
st.success("If you can see this, the app is running correctly!")

# Simple interaction
name = st.text_input("Enter your name:", value="Test User")
if name:
    st.balloons()  # Visual confirmation
    st.write(f"👋 Hello **{name}**!")

# Show some data
import pandas as pd
import numpy as np

st.subheader("📊 Sample Data")
data = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randn(5)
})
st.dataframe(data)

# Add sidebar
st.sidebar.title("📋 Sidebar")
st.sidebar.success("✅ Sidebar working!")

# Footer with debug info
st.markdown("---")
st.write(f"🐍 Python: Streamlit {st.__version__}")
st.write("🎯 **Status: FULLY OPERATIONAL**")










