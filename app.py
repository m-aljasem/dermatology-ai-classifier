"""Streamlit app for skin cancer classification"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

from src.explainability import ModelExplainer
from src.model import build_skin_lesion_cnn
from src.abcde_analysis import ABCDEAnalyzer
import tensorflow as tf

# Cache the model for better performance (especially on Streamlit Cloud)
@st.cache_resource
def load_model_cached(model_path):
    """Load and cache the model."""
    if model_path.exists():
        try:
            return tf.keras.models.load_model(str(model_path))
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Page configuration
st.set_page_config(
    page_title="Dermatology AI Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #d32f2f;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-moderate {
        background-color: #fff3e0;
        border-left-color: #f57c00;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #388e3c;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .tool-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 1rem 0;
    }
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
    </style>
""", unsafe_allow_html=True)

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma - precancerous lesions',
    'bcc': 'Basal cell carcinoma - most common skin cancer, rarely metastasizes',
    'bkl': 'Benign keratosis-like lesions - non-cancerous growths',
    'df': 'Dermatofibroma - benign fibrous skin lesion',
    'mel': 'Melanoma - most dangerous form of skin cancer',
    'nv': 'Melanocytic nevi - common moles, usually benign',
    'vasc': 'Vascular lesions - blood vessel abnormalities'
}

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "skin_cancer_model.keras"

def render_footer():
    """Render footer with author and project information."""
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

# Main header
st.markdown('<div class="main-header">🩺 Dermatology AI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Skin Lesion Classification & Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.info("""
    **How to use:**
    1. Upload a dermatoscopic image
    2. Click 'Classify' for AI prediction
    3. Use 'Risk Assessment' for ABCDE analysis
    4. View explainability visualizations
    """)
    
    st.divider()
    st.markdown("**Model Status:**")
    if WEIGHTS_PATH.exists():
        model_size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
        st.success(f"✅ Model loaded ({model_size_mb:.1f} MB)")
    else:
        st.error("❌ Model not found")
        with st.expander("📥 How to get the model"):
            st.markdown("""
            **If you cloned the repository:**
            
            The model is included via Git LFS. Run:
            ```bash
            git lfs install
            git lfs pull
            ```
            
            **If Git LFS is not available:**
            
            **Option 1: Download from GitHub Releases**
            1. Go to [GitHub Releases](https://github.com/m-aljasem/dermatology-ai-classifier/releases)
            2. Download `skin_cancer_model.keras`
            3. Place it in the `models/` directory
            
            **Option 2: Use download script**
            ```bash
            python scripts/download_model.py
            ```
            
            **Option 3: Train your own**
            Run `notebooks/02_model_training.ipynb` on Kaggle
            """)
    
    st.divider()
    st.markdown("**📊 About**")
    st.markdown("Visit the About page from the sidebar to learn more about this project.")
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dermatoscopic image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin lesion for analysis"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).resize((250, 250))
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Clinical features input (for ABCDE analysis)
            st.subheader("📏 Clinical Features (Optional)")
            with st.expander("Add clinical information for enhanced risk assessment"):
                diameter_mm = st.number_input(
                    "Diameter (mm)",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.1,
                    help="Measure the largest diameter of the lesion"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    size_change = st.number_input(
                        "Size change (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        help="Percentage increase in size over time"
                    )
                with col_b:
                    color_change = st.checkbox("Color change", help="Has the lesion changed color?")
                    shape_change = st.checkbox("Shape change", help="Has the lesion changed shape?")
            
            img_array = np.array(img) / 255.0
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            img_array = np.expand_dims(img_array, 0)
            
            # Classification button
            classify_disabled = not WEIGHTS_PATH.exists()
            if st.button(
                "🔍 Classify Lesion", 
                type="primary", 
                use_container_width=True,
                disabled=classify_disabled
            ):
                if not WEIGHTS_PATH.exists():
                    st.error("""
                    **Model not found!**
                    
                    Please download the model first. See the sidebar for instructions.
                    """)
                    st.stop()
                
                with st.spinner("Loading model and analyzing image..."):
                    # Ensure models directory exists
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)

                    # Load model (cached for performance)
                    model = load_model_cached(WEIGHTS_PATH)
                    
                    if model is None:
                        if not WEIGHTS_PATH.exists():
                            st.error("""
                            **Model file not found!**
                            
                            The model file `models/skin_cancer_model.keras` is required for predictions.
                            
                            **If deploying on Streamlit Cloud:**
                            - Ensure the model is committed via Git LFS
                            - Streamlit Cloud will automatically pull LFS files
                            
                            **If running locally:**
                            1. Check the sidebar for download instructions
                            2. Or run: `python scripts/download_model.py`
                            3. Then refresh this page
                            """)
                        st.stop()
                    
                    # Store model in session state
                    st.session_state['model'] = model
                    st.session_state['img_array'] = img_array
                    
                    # Make prediction
                    pred = model.predict(img_array, verbose=0)
                    class_idx = np.argmax(pred[0])
                    class_name = CLASSES[class_idx]
                    confidence = pred[0][class_idx]
                    
                    # Store results in session state
                    st.session_state['prediction'] = {
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'confidence': confidence,
                        'probabilities': pred[0]
                    }
            
            # Display results if available
            if 'prediction' in st.session_state:
                pred_data = st.session_state['prediction']
                
                st.divider()
                st.header("🎯 Classification Results")
                
                # Main prediction card
                col_pred, col_conf = st.columns([2, 1])
                with col_pred:
                    st.markdown(f"### {CLASS_NAMES[pred_data['class_name']]}")
                    st.caption(CLASS_DESCRIPTIONS[pred_data['class_name']])
                with col_conf:
                    st.metric("Confidence", f"{pred_data['confidence']:.1%}")
                
                # All probabilities
                with st.expander("📊 View All Probabilities"):
                    prob_data = {
                        CLASS_NAMES[cls]: float(prob)
                        for cls, prob in zip(CLASSES, pred_data['probabilities'])
                    }
                    st.bar_chart(prob_data)
    
    with col2:
        st.header("🛠️ Analysis Tools")
        
        # Risk Assessment Tool
        st.markdown('<div class="tool-section">', unsafe_allow_html=True)
        st.markdown("### 🔬 ABCDE Risk Assessment")
        st.markdown("Analyze lesions using the ABCDE rule for melanoma detection")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if 'img_array' in st.session_state and st.button("📈 Perform Risk Assessment", use_container_width=True):
            with st.spinner("Analyzing lesion characteristics..."):
                img_array = st.session_state['img_array']
                img_for_analysis = (img_array[0] * 255).astype(np.uint8)
                
                # Get clinical features
                clinical_features = {}
                if 'diameter_mm' in locals() and diameter_mm > 0:
                    clinical_features['diameter_mm'] = diameter_mm
                if 'size_change' in locals() or 'color_change' in locals() or 'shape_change' in locals():
                    evolution = {}
                    if 'size_change' in locals() and size_change > 0:
                        evolution['size_change_percent'] = size_change
                    if 'color_change' in locals():
                        evolution['color_change'] = color_change
                    if 'shape_change' in locals():
                        evolution['shape_change'] = shape_change
                    if evolution:
                        clinical_features['evolution'] = evolution
                
                # Perform ABCDE analysis
                analyzer = ABCDEAnalyzer()
                analysis = analyzer.analyze_lesion(img_for_analysis, clinical_features if clinical_features else None)
                
                # Store in session state
                st.session_state['risk_analysis'] = analysis
                
                # Display results
                st.success("✅ Risk assessment completed!")
                
                # Risk level display
                risk_level = analysis['risk_level']
                risk_score = analysis['risk_score']
                
                if risk_level == "High":
                    st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                    st.markdown(f"### ⚠️ {risk_level} Risk")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif risk_level in ["Moderate-High", "Moderate"]:
                    st.markdown(f'<div class="risk-moderate">', unsafe_allow_html=True)
                    st.markdown(f"### ⚠️ {risk_level} Risk")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                    st.markdown(f"### ✓ {risk_level} Risk")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk score metric
                st.metric("Risk Score", f"{risk_score}/100")
                
                # ABCDE scores
                st.subheader("📋 ABCDE Scores")
                scores = analysis['abcde_scores']
                for criterion, score in scores.items():
                    st.progress(score, text=f"{criterion}: {score:.2f}")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                # Interpretation
                st.info(f"**Interpretation:** {analysis['interpretation']}")
        
        elif 'risk_analysis' in st.session_state:
            # Display cached risk analysis
            analysis = st.session_state['risk_analysis']
            risk_level = analysis['risk_level']
            risk_score = analysis['risk_score']
            
            if risk_level == "High":
                st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ {risk_level} Risk")
                st.markdown('</div>', unsafe_allow_html=True)
            elif risk_level in ["Moderate-High", "Moderate"]:
                st.markdown(f'<div class="risk-moderate">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ {risk_level} Risk")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                st.markdown(f"### ✓ {risk_level} Risk")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.metric("Risk Score", f"{risk_score}/100")
            
            st.subheader("📋 ABCDE Scores")
            scores = analysis['abcde_scores']
            for criterion, score in scores.items():
                st.progress(score, text=f"{criterion}: {score:.2f}")
            
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        st.divider()
        
        # Explainability Tool
        st.markdown('<div class="tool-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Explainability Analysis")
        st.markdown("Visualize which parts of the image influenced the prediction")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state and 'model' in st.session_state:
            if st.button("🎨 Generate SHAP Visualization", use_container_width=True):
                try:
                    import matplotlib.pyplot as plt
                    import shap
                    
                    model = st.session_state['model']
                    img_array = st.session_state['img_array']
                    pred_data = st.session_state['prediction']
                    
                    with st.spinner("Generating explainability visualization..."):
                        explainer = ModelExplainer(model, img_array[:1])
                        shap_values = explainer.explain_instance(
                            img_array, 
                            plot=False, 
                            class_idx=pred_data['class_idx']
                        )
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        axes[0].imshow(img_array[0])
                        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
                        axes[0].axis('off')
                        
                        shap_image = np.abs(shap_values[0]).sum(axis=2) if len(shap_values[0].shape) == 3 else np.abs(shap_values[0])
                        axes[1].imshow(shap_image, cmap='hot')
                        axes[1].set_title('SHAP Values (Importance)', fontsize=12, fontweight='bold')
                        axes[1].axis('off')
                        
                        axes[2].imshow(img_array[0])
                        axes[2].imshow(shap_image, cmap='hot', alpha=0.5)
                        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.caption("Red areas indicate regions that strongly influenced the prediction")
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    st.info("Make sure SHAP is installed: `pip install shap`")
        else:
            st.info("👆 Upload an image and classify it first to use explainability tools")
    
    # Footer
    render_footer()
