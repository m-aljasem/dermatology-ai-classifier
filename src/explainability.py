"""
Explainability utilities for skin-cancer using SHAP and other interpretability tools.

This module provides model interpretability features crucial for medical AI applications,
including SHAP values, feature importance, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

# LIME imports (for image explanations)
try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")


class ModelExplainer:
    """
    Explainability wrapper for TensorFlow/Keras models.
    Uses GradientExplainer for deep learning models.
    """
    
    def __init__(self, model, X_train_sample=None, feature_names=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained Keras model
            X_train_sample: Sample of training data (for background)
            feature_names: List of feature names (for tabular data)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        
        # For image models, use GradientExplainer
        if X_train_sample is not None:
            # Use subset for faster computation
            if len(X_train_sample) > 100:
                X_train_sample = X_train_sample[:100]
            self.explainer = shap.GradientExplainer(model, X_train_sample)
        else:
            self.explainer = None
    
    def explain_instance(self, instance, plot=True, class_idx=None):
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain
            plot: Whether to plot SHAP values
            class_idx: Class index to explain (for multi-class)
            
        Returns:
            shap_values: SHAP values
        """
        if len(instance.shape) == 3:  # Single image
            instance = np.expand_dims(instance, 0)
        
        if self.explainer is None:
            # Create explainer on-the-fly
            self.explainer = shap.GradientExplainer(self.model, instance)
        
        shap_values = self.explainer.shap_values(instance)
        
        # Handle multi-output models
        if isinstance(shap_values, list):
            if class_idx is not None:
                shap_values = shap_values[class_idx]
            else:
                shap_values = shap_values[0]  # Default to first class
        
        if plot and len(instance.shape) == 4:  # Image data
            # Plot original image and SHAP heatmap
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(instance[0])
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # SHAP values
            shap_image = shap_values[0]
            if len(shap_image.shape) == 3:
                shap_image = np.abs(shap_image).sum(axis=2)  # Sum across channels
            
            axes[1].imshow(shap_image, cmap='hot')
            axes[1].set_title('SHAP Values (Importance)')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(instance[0])
            axes[2].imshow(shap_image, cmap='hot', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return shap_values
    
    def explain_dataset(self, X, max_instances=50, plot=True, class_idx=None):
        """
        Explain multiple instances.
        
        Args:
            X: Instances to explain
            max_instances: Maximum number of instances
            plot: Whether to plot summary
            class_idx: Class index to explain
            
        Returns:
            shap_values: SHAP values
        """
        if len(X) > max_instances:
            X = X[:max_instances]
        
        if self.explainer is None:
            # Create explainer with sample
            sample = X[:min(10, len(X))]
            self.explainer = shap.GradientExplainer(self.model, sample)
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            if class_idx is not None:
                shap_values = shap_values[class_idx]
            else:
                shap_values = shap_values[0]
        
        if plot and len(X.shape) == 4:  # Image data
            # Plot summary
            shap.image_plot(shap_values, X, show=False)
            plt.tight_layout()
            plt.show()
        
        return shap_values
    
    def get_feature_importance(self, X_sample, class_idx=None):
        """
        Get global feature importance.
        
        Args:
            X_sample: Sample data
            class_idx: Class index
            
        Returns:
            importance: Average absolute SHAP values
        """
        shap_values = self.explain_dataset(X_sample, max_instances=50, plot=False, class_idx=class_idx)
        importance = np.abs(shap_values).mean(axis=0)
        return importance


def create_lime_explainer(model, preprocess_fn=None):
    """
    Create LIME explainer for image models.
    
    Args:
        model: Trained model
        preprocess_fn: Optional preprocessing function
        
    Returns:
        explainer: LIME explainer
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explainer = lime_image.LimeImageExplainer()
    return explainer


def explain_with_lime(explainer, image, model, top_labels=5, num_features=10):
    """
    Explain image prediction using LIME.
    
    Args:
        explainer: LIME explainer
        image: Input image
        model: Trained model
        top_labels: Number of top labels to explain
        num_features: Number of features to show
        
    Returns:
        explanation: LIME explanation
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explanation = explainer.explain_instance(
        image.astype('double'),
        model.predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation


def plot_lime_explanation(explanation, label=1, figsize=(10, 5)):
    """
    Plot LIME explanation.
    
    Args:
        explanation: LIME explanation object
        label: Label to explain
        figsize: Figure size
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(temp)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask)
    axes[1].set_title('LIME Explanation (Important Regions)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
