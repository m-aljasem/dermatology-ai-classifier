"""Professional Streamlit app for skin cancer classification"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from src.model import build_skin_lesion_cnn
from src.abcde_analysis import ABCDEAnalyzer
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Dermatology AI Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #5f6368;
        text-align: center;
        padding-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1a73e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(238, 90, 111, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(251, 140, 0, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(67, 160, 71, 0.3);
    }
    
    /* Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 2px solid #e8eaed;
        text-align: center;
        color: #5f6368;
        font-size: 0.9rem;
    }
    
    .footer a {
        color: #1a73e8;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    /* Metric cards */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
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
    'akiec': 'Actinic keratoses and intraepithelial carcinoma - precancerous lesions that may develop into skin cancer.',
    'bcc': 'Basal cell carcinoma - most common form of skin cancer, rarely metastasizes but can be locally destructive.',
    'bkl': 'Benign keratosis-like lesions - non-cancerous growths that are typically harmless.',
    'df': 'Dermatofibroma - benign fibrous skin lesion, usually harmless.',
    'mel': 'Melanoma - most dangerous form of skin cancer, requires immediate medical attention.',
    'nv': 'Melanocytic nevi - common moles, usually benign but should be monitored.',
    'vasc': 'Vascular lesions - abnormalities of blood vessels, usually benign.'
}

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "skin_cancer_model.keras"

# Cache model loading
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

def render_footer():
    """Render professional footer."""
    st.markdown("""
    <div class="footer">
        <p><strong>Developed by</strong> <a href="https://aljasem.eu.org" target="_blank">Mohamad AlJasem, MD MPH MSc</a></p>
        <p><strong>Project Code:</strong> <a href="https://github.com/m-aljasem/dermatology-ai-classifier" target="_blank">GitHub Repository</a></p>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: #999;">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            Not intended for clinical diagnosis. Always consult a qualified healthcare professional.
        </p>
    </div>
    """, unsafe_allow_html=True)

def safe_shap_explanation(model, img_array, class_idx):
    """Generate SHAP explanation with proper error handling."""
    try:
        import shap
        
        # Create a background dataset (use the image itself as background)
        background = img_array[:1]  # Use single image as background
        
        # Use GradientExplainer (more reliable for this use case)
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(img_array)
        except Exception as e1:
            # Fallback to DeepExplainer
            try:
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(img_array)
            except Exception as e2:
                return None, f"SHAP explainer failed: {str(e1)}; {str(e2)}"
        
        # Handle multi-class output - SHAP returns list of arrays for each class
        if isinstance(shap_values, list):
            # Select the SHAP values for the predicted class
            if class_idx < len(shap_values):
                shap_values = shap_values[class_idx]
            else:
                shap_values = shap_values[0]  # Fallback to first class
        
        # Process SHAP values for visualization
        # Handle different possible shapes
        original_shape = shap_values.shape
        
        # Case 1: 5D shape (batch, height, width, channels, classes) - unexpected but handle it
        if len(shap_values.shape) == 5:
            if class_idx < shap_values.shape[4]:
                shap_values = shap_values[0, :, :, :, class_idx]  # (height, width, channels)
            else:
                shap_values = shap_values[0, :, :, :, 0]  # Fallback to first class
        
        # Case 2: 4D shape (batch, height, width, channels)
        elif len(shap_values.shape) == 4:
            shap_values = shap_values[0]  # Take first batch -> (height, width, channels)
        
        # Case 3: 3D shape (height, width, channels) - already correct
        elif len(shap_values.shape) == 3:
            pass  # Already in correct format
        
        # Case 4: 2D shape (height, width) - already processed
        elif len(shap_values.shape) == 2:
            return shap_values, None
        
        else:
            return None, f"Unexpected SHAP shape: {original_shape}"
        
        # Ensure we have 3D shape (height, width, channels) at this point
        if len(shap_values.shape) != 3:
            return None, f"SHAP values must be 3D after processing, got shape: {shap_values.shape}"
        
        # Sum across channels to get 2D heatmap
        shap_image = np.abs(shap_values).sum(axis=2)
        
        # Ensure it's 2D
        if len(shap_image.shape) != 2:
            return None, f"SHAP image must be 2D, got shape: {shap_image.shape}"
        
        return shap_image, None
    except Exception as e:
        import traceback
        return None, f"SHAP explanation error: {str(e)}\n{traceback.format_exc()}"

# ==================== MAIN APP ====================

# Header
st.markdown('<div class="main-header">🩺 Dermatology AI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Skin Lesion Classification & Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar - Minimal information only
with st.sidebar:
    st.header("ℹ️ Quick Info")
    
    # Model status
    if WEIGHTS_PATH.exists():
        model_size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
        st.success(f"✅ Model Ready\n({model_size_mb:.1f} MB)")
    else:
        st.error("❌ Model Missing")
        st.info("See deployment guide for setup instructions")
    
    st.divider()
    
    st.markdown("**How to Use:**")
    st.markdown("""
    1. Upload lesion image
    2. Click Classify
    3. Review results
    4. Use risk assessment
    """)
    
    st.divider()
    st.markdown("**📊 About**")
    st.markdown("Visit the About page for detailed information.")

# Main content area
tab1, tab2 = st.tabs(["🔍 Classification", "📊 Risk Assessment"])

# ==================== TAB 1: CLASSIFICATION ====================
with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.header("📤 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a dermatoscopic image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, well-lit image of the skin lesion",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).resize((250, 250))
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Convert to array
            img_array = np.array(img) / 255.0
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            img_array = np.expand_dims(img_array, 0)
            
            # Store in session state
            st.session_state['img_array'] = img_array
            st.session_state['uploaded_image'] = img
            
            # Classification button
            classify_disabled = not WEIGHTS_PATH.exists()
            
            if st.button("🔍 Classify Lesion", type="primary", use_container_width=True, disabled=classify_disabled):
                if not WEIGHTS_PATH.exists():
                    st.error("Model file not found. Please ensure the model is properly set up.")
                    st.stop()
                
                with st.spinner("🔄 Analyzing image with AI model..."):
                    model = load_model_cached(WEIGHTS_PATH)
                    
                    if model is None:
                        st.error("Failed to load model. Please check the model file.")
                        st.stop()
                    
                    # Store model
                    st.session_state['model'] = model
                    
                    # Make prediction
                    pred = model.predict(img_array, verbose=0)
                    class_idx = np.argmax(pred[0])
                    class_name = CLASSES[class_idx]
                    confidence = pred[0][class_idx]
                    
                    # Store results
                    st.session_state['prediction'] = {
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'confidence': confidence,
                        'probabilities': pred[0]
                    }
                    
                    st.success("✅ Classification complete!")
    
    with col2:
        st.header("🎯 Results")
        
        if 'prediction' in st.session_state:
            pred_data = st.session_state['prediction']
            
            # Main prediction card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Class name and confidence
            col_name, col_conf = st.columns([2, 1])
            with col_name:
                st.markdown(f"### {CLASS_NAMES[pred_data['class_name']]}")
            with col_conf:
                st.metric("Confidence", f"{pred_data['confidence']:.1%}")
            
            # Description
            st.caption(CLASS_DESCRIPTIONS[pred_data['class_name']])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability distribution
            st.subheader("📊 Probability Distribution")
            prob_data = {
                CLASS_NAMES[cls]: float(prob)
                for cls, prob in zip(CLASSES, pred_data['probabilities'])
            }
            
            # Sort by probability
            sorted_probs = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(sorted_probs)
            
            # Top 3 predictions
            st.subheader("🏆 Top Predictions")
            sorted_items = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (name, prob) in enumerate(sorted_items, 1):
                medal = ["🥇", "🥈", "🥉"][i-1]
                st.write(f"{medal} **{name}**: {prob:.1%}")
            
            # Explainability section
            st.divider()
            st.subheader("🔍 Explainability")
            
            if st.button("🎨 Generate Visualization", use_container_width=True):
                if 'model' in st.session_state and 'img_array' in st.session_state:
                    with st.spinner("Generating explainability visualization..."):
                        model = st.session_state['model']
                        img_array = st.session_state['img_array']
                        class_idx = pred_data['class_idx']
                        
                        shap_image, error = safe_shap_explanation(model, img_array, class_idx)
                        
                        if shap_image is not None:
                            # Create visualization
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            # Original image
                            axes[0].imshow(img_array[0])
                            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
                            axes[0].axis('off')
                            
                            # SHAP heatmap
                            im = axes[1].imshow(shap_image, cmap='hot')
                            axes[1].set_title('SHAP Values (Importance)', fontsize=12, fontweight='bold')
                            axes[1].axis('off')
                            plt.colorbar(im, ax=axes[1], fraction=0.046)
                            
                            # Overlay
                            axes[2].imshow(img_array[0])
                            axes[2].imshow(shap_image, cmap='hot', alpha=0.5)
                            axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
                            axes[2].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            st.caption("🔴 Red areas indicate regions that strongly influenced the prediction")
                        else:
                            st.warning(f"Could not generate SHAP visualization: {error}")
                            st.info("""
                            **Alternative:** The model prediction is based on learned patterns from thousands of training images.
                            The classification considers features like asymmetry, border irregularity, color variation, and texture.
                            """)
                else:
                    st.info("Please classify an image first to generate visualizations.")
        else:
            st.info("👆 Upload an image and click 'Classify' to see results here.")

# ==================== TAB 2: RISK ASSESSMENT ====================
with tab2:
    st.header("🔬 ABCDE Risk Assessment")
    st.markdown("Analyze skin lesions using the ABCDE rule for melanoma detection")
    
    if 'img_array' not in st.session_state:
        st.warning("⚠️ Please upload and classify an image first in the Classification tab.")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📏 Clinical Features")
        st.markdown("Provide additional clinical information for enhanced risk assessment:")
        
        diameter_mm = st.number_input(
            "Diameter (mm)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
            help="Measure the largest diameter of the lesion"
        )
        
        st.markdown("**Evolution (Changes over time):**")
        col_a, col_b = st.columns(2)
        with col_a:
            size_change = st.number_input(
                "Size change (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="Percentage increase in size"
            )
        with col_b:
            color_change = st.checkbox("Color change", help="Has the lesion changed color?")
            shape_change = st.checkbox("Shape change", help="Has the lesion changed shape?")
        
        if st.button("📈 Perform Risk Assessment", type="primary", use_container_width=True):
            with st.spinner("Analyzing lesion characteristics..."):
                img_array = st.session_state['img_array']
                img_for_analysis = (img_array[0] * 255).astype(np.uint8)
                
                # Prepare clinical features
                clinical_features = {}
                if diameter_mm > 0:
                    clinical_features['diameter_mm'] = diameter_mm
                
                evolution = {}
                if size_change > 0:
                    evolution['size_change_percent'] = size_change
                if color_change:
                    evolution['color_change'] = True
                if shape_change:
                    evolution['shape_change'] = True
                if evolution:
                    clinical_features['evolution'] = evolution
                
                # Perform ABCDE analysis
                analyzer = ABCDEAnalyzer()
                analysis = analyzer.analyze_lesion(
                    img_for_analysis, 
                    clinical_features if clinical_features else None
                )
                
                # Store in session state
                st.session_state['risk_analysis'] = analysis
                st.success("✅ Risk assessment completed!")
    
    with col2:
        st.subheader("📊 Assessment Results")
        
        if 'risk_analysis' in st.session_state:
            analysis = st.session_state['risk_analysis']
            risk_level = analysis['risk_level']
            risk_score = analysis['risk_score']
            
            # Risk level display
            if risk_level == "High":
                st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ {risk_level} Risk")
                st.markdown(f"**Risk Score: {risk_score}/100**")
                st.markdown('</div>', unsafe_allow_html=True)
            elif risk_level in ["Moderate-High", "Moderate"]:
                st.markdown(f'<div class="risk-moderate">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ {risk_level} Risk")
                st.markdown(f"**Risk Score: {risk_score}/100**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                st.markdown(f"### ✓ {risk_level} Risk")
                st.markdown(f"**Risk Score: {risk_score}/100**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ABCDE scores
            st.subheader("📋 ABCDE Scores")
            scores = analysis['abcde_scores']
            for criterion, score in scores.items():
                st.progress(score, text=f"{criterion}: {score:.2f}")
            
            # Interpretation
            st.info(f"**Interpretation:** {analysis['interpretation']}")
            
            # Recommendations
            st.subheader("💡 Clinical Recommendations")
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("👆 Click 'Perform Risk Assessment' to see results here.")
            st.markdown("""
            **ABCDE Rule:**
            - **A** - Asymmetry
            - **B** - Border irregularity
            - **C** - Color variation
            - **D** - Diameter (>6mm)
            - **E** - Evolution (changes)
            """)

# Footer
st.divider()
render_footer()
