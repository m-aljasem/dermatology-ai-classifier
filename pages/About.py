"""About page for Dermatology AI Classifier"""
import streamlit as st

st.set_page_config(
    page_title="About - Dermatology AI Classifier",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 2px solid #e0e0e0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }
    .footer a {
        color: #1f77b4;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 About This Project")

st.markdown("""
## 🎯 Project Overview

This **Dermatology AI Classifier** is an advanced deep learning system designed to assist 
healthcare professionals in the classification and risk assessment of skin lesions. The system 
leverages state-of-the-art convolutional neural networks trained on the HAM10000 dataset to 
classify dermatoscopic images into 7 distinct lesion types.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✨ Key Features
    
    #### 🤖 AI Classification
    - **7-Class Classification**: Accurately identifies 7 types of skin lesions
    - **Confidence Scoring**: Provides probability scores for each class
    - **High Accuracy**: Trained on thousands of dermatoscopic images
    
    #### 🔬 ABCDE Risk Assessment
    - **Asymmetry Analysis**: Evaluates lesion symmetry
    - **Border Irregularity**: Detects irregular borders
    - **Color Variation**: Analyzes color diversity
    - **Diameter Assessment**: Considers lesion size
    - **Evolution Tracking**: Monitors changes over time
    - **Risk Scoring**: Provides comprehensive risk assessment
    """)

with col2:
    st.markdown("""
    #### 🔍 Explainability
    - **SHAP Visualization**: Shows which image regions influence predictions
    - **Transparent AI**: Understand how the model makes decisions
    - **Clinical Insights**: Helps clinicians understand AI reasoning
    
    #### 📊 Advanced Analytics
    - **Probability Distributions**: View confidence across all classes
    - **Visual Interpretations**: Heatmaps and overlays
    - **Clinical Recommendations**: Evidence-based guidance
    """)

st.divider()

st.markdown("""
## 🧬 Classified Lesion Types

1. **akiec** - Actinic keratoses and intraepithelial carcinoma
2. **bcc** - Basal cell carcinoma
3. **bkl** - Benign keratosis-like lesions
4. **df** - Dermatofibroma
5. **mel** - Melanoma
6. **nv** - Melanocytic nevi
7. **vasc** - Vascular lesions
""")

st.divider()

st.markdown("""
## 🛠️ Technical Details

- **Model Architecture**: Custom CNN with Batch Normalization
- **Input Resolution**: 250×250 RGB images
- **Framework**: TensorFlow/Keras
- **Training Dataset**: HAM10000 (Human Against Machine with 10,000 training images)
- **Model Format**: `.keras` (modern TensorFlow format)
- **Training**: GPU-accelerated with advanced callbacks and data augmentation
""")

st.divider()

st.markdown("""
## 📚 Methodology

The model uses a custom convolutional neural network architecture optimized for dermatoscopic 
image analysis. Training includes:

- Aggressive data augmentation (rotation, flipping, zooming, brightness)
- Stratified train/validation split
- Advanced callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- Comprehensive evaluation metrics
- GPU-optimized training pipeline
""")

st.divider()

st.markdown("""
## ⚠️ Important Disclaimer

**This tool is for research and educational purposes only.**

- Not intended for clinical diagnosis
- Should not replace professional medical judgment
- Always consult qualified healthcare professionals
- Results should be interpreted by trained dermatologists
- The AI model is a decision support tool, not a replacement for clinical expertise
""")

st.divider()

st.markdown("""
## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{dermatology-ai-classifier,
  title = {Dermatology AI Classifier},
  author = {AlJasem, Mohamad},
  year = {2024},
  url = {https://github.com/m-aljasem/dermatology-ai-classifier}
}
```
""")

st.divider()

st.markdown("""
## 🔗 Resources

- **Dataset**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Documentation**: See project README for detailed technical documentation
- **Code**: Available on GitHub (see footer below)
""")

st.divider()

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Developed by</strong> <a href="https://aljasem.eu.org" target="_blank">Mohamad AlJasem, MD MPH MSc</a></p>
    <p><strong>Project Code:</strong> <a href="https://github.com/m-aljasem/dermatology-ai-classifier" target="_blank">https://github.com/m-aljasem/dermatology-ai-classifier</a></p>
    <p style="margin-top: 1rem; font-size: 0.85rem; color: #999;">
        ⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
        Not intended for clinical diagnosis. Always consult a qualified healthcare professional.
    </p>
</div>
""", unsafe_allow_html=True)

