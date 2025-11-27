"""
ABCDE Analysis for Skin Lesion Classification

Implements the ABCDE rule for melanoma detection:
- Asymmetry
- Border irregularity
- Color variation
- Diameter
- Evolution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Use cv2 with fallback for environments without GUI libraries
# Handle both ImportError and OSError (libGL.so.1 missing)
cv2 = None
try:
    import cv2
    # Test if cv2 actually works (might fail at runtime due to missing libGL.so.1)
    _test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.cvtColor(_test_img, cv2.COLOR_RGB2GRAY)
except (ImportError, OSError, AttributeError) as e:
    # If opencv-python fails, try to use opencv-python-headless
    try:
        import sys
        import subprocess
        import importlib
        # Try importing opencv-python-headless
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--quiet", "--disable-pip-version-check"], 
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        cv2 = importlib.import_module('cv2')
    except:
        # If all else fails, create a mock cv2 for basic operations
        class MockCV2:
            COLOR_RGB2GRAY = 7
            COLOR_RGB2HSV = 40
            RETR_EXTERNAL = 0
            CHAIN_APPROX_SIMPLE = 2
            @staticmethod
            def cvtColor(img, code):
                if code == 7:  # COLOR_RGB2GRAY
                    if len(img.shape) == 3:
                        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                    return img.astype(np.uint8)
                elif code == 40:  # COLOR_RGB2HSV
                    # Simple RGB to HSV conversion
                    img_float = img.astype(np.float32) / 255.0
                    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]
                    max_val = np.maximum(np.maximum(r, g), b)
                    min_val = np.minimum(np.minimum(r, g), b)
                    diff = max_val - min_val
                    h = np.zeros_like(max_val)
                    h[max_val == r] = (60 * ((g[max_val == r] - b[max_val == r]) / diff[max_val == r]) + 360) % 360
                    h[max_val == g] = (60 * ((b[max_val == g] - r[max_val == g]) / diff[max_val == g]) + 120) % 360
                    h[max_val == b] = (60 * ((r[max_val == b] - g[max_val == b]) / diff[max_val == b]) + 240) % 360
                    s = np.where(max_val == 0, 0, diff / max_val)
                    v = max_val
                    return np.stack([h, s * 255, v * 255], axis=-1).astype(np.uint8)
                return img
            @staticmethod
            def Canny(img, low, high):
                # Simple edge detection fallback using scipy
                try:
                    from scipy import ndimage
                    sobel_x = ndimage.sobel(img.astype(float), axis=1)
                    sobel_y = ndimage.sobel(img.astype(float), axis=0)
                    edges = np.hypot(sobel_x, sobel_y)
                    threshold = (low + high) / 2
                    return (edges > threshold).astype(np.uint8) * 255
                except:
                    # Fallback without scipy
                    return np.zeros_like(img, dtype=np.uint8)
            @staticmethod
            def findContours(img, mode, method):
                # Simple contour finding fallback
                return [], None
            @staticmethod
            def arcLength(contour, closed):
                if len(contour) < 2:
                    return 0.0
                try:
                    contour_2d = contour.reshape(-1, 2) if len(contour.shape) > 2 else contour
                    diffs = np.diff(contour_2d, axis=0)
                    return np.sum(np.linalg.norm(diffs, axis=1))
                except:
                    return 0.0
            @staticmethod
            def contourArea(contour):
                if len(contour) < 3:
                    return 0.0
                try:
                    # Shoelace formula
                    if len(contour.shape) > 2:
                        x = contour[:, 0, 0]
                        y = contour[:, 0, 1]
                    else:
                        x = contour[:, 0]
                        y = contour[:, 1]
                    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                except:
                    return 0.0
        cv2 = MockCV2()


class ABCDEAnalyzer:
    """Analyze skin lesions using ABCDE criteria."""
    
    def __init__(self):
        """Initialize ABCDE analyzer."""
        self.criteria_weights = {
            'asymmetry': 0.2,
            'border': 0.2,
            'color': 0.2,
            'diameter': 0.2,
            'evolution': 0.2
        }
    
    def analyze_lesion(self, lesion_image: np.ndarray, 
                      clinical_features: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive ABCDE analysis.
        
        Args:
            lesion_image: Lesion image (numpy array)
            clinical_features: Optional clinical features (diameter, evolution, etc.)
            
        Returns:
            ABCDE analysis results
        """
        # Convert to grayscale if needed
        try:
            if len(lesion_image.shape) == 3:
                gray_image = cv2.cvtColor(lesion_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = lesion_image
        except Exception:
            # Fallback: manual grayscale conversion
            if len(lesion_image.shape) == 3:
                gray_image = np.dot(lesion_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray_image = lesion_image
        
        # Analyze each criterion
        asymmetry_score = self._analyze_asymmetry(gray_image)
        border_score = self._analyze_border(gray_image)
        color_score = self._analyze_color(lesion_image) if len(lesion_image.shape) == 3 else 0
        diameter_score = self._analyze_diameter(clinical_features) if clinical_features else 0
        evolution_score = self._analyze_evolution(clinical_features) if clinical_features else 0
        
        # Calculate total score
        total_score = (
            asymmetry_score * self.criteria_weights['asymmetry'] +
            border_score * self.criteria_weights['border'] +
            color_score * self.criteria_weights['color'] +
            diameter_score * self.criteria_weights['diameter'] +
            evolution_score * self.criteria_weights['evolution']
        )
        
        # Risk assessment
        risk_assessment = self._assess_risk(total_score)
        
        return {
            'abcde_scores': {
                'A - Asymmetry': round(asymmetry_score, 2),
                'B - Border': round(border_score, 2),
                'C - Color': round(color_score, 2),
                'D - Diameter': round(diameter_score, 2),
                'E - Evolution': round(evolution_score, 2)
            },
            'total_score': round(total_score, 2),
            'risk_level': risk_assessment['risk_level'],
            'risk_score': risk_assessment['risk_score'],
            'recommendations': risk_assessment['recommendations'],
            'interpretation': self._interpret_scores({
                'asymmetry': asymmetry_score,
                'border': border_score,
                'color': color_score,
                'diameter': diameter_score,
                'evolution': evolution_score
            })
        }
    
    def _analyze_asymmetry(self, image: np.ndarray) -> float:
        """
        Analyze asymmetry (A).
        Score: 0-1, higher = more asymmetric
        """
        # Divide image into quadrants and compare
        h, w = image.shape
        mid_h, mid_w = h // 2, w // 2
        
        # Compare opposite quadrants
        q1 = image[:mid_h, :mid_w]
        q3 = image[mid_h:, mid_w:]
        
        q2 = image[:mid_h, mid_w:]
        q4 = image[mid_h:, :mid_w]
        
        # Calculate differences
        diff1 = np.abs(q1 - np.flip(q3)).mean()
        diff2 = np.abs(q2 - np.flip(q4)).mean()
        
        avg_diff = (diff1 + diff2) / 2
        max_diff = image.max() - image.min()
        
        # Normalize to 0-1
        asymmetry_score = min(1.0, avg_diff / max_diff if max_diff > 0 else 0)
        
        return asymmetry_score
    
    def _analyze_border(self, image: np.ndarray) -> float:
        """
        Analyze border irregularity (B).
        Score: 0-1, higher = more irregular
        """
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Find contours
            contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Handle different OpenCV versions (returns 2 or 3 values)
            if len(contours_result) == 3:
                _, contours, _ = contours_result
            else:
                contours, _ = contours_result
            
            if not contours or len(contours) == 0:
                return 0.0
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate perimeter and area
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            
            # Circularity measure (irregular borders have lower circularity)
            if area > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                border_score = 1.0 - circularity  # Inverse: lower circularity = higher score
            else:
                border_score = 0.0
            
            return min(1.0, border_score)
        except Exception as e:
            # Fallback: simple border analysis without OpenCV
            # Use gradient-based edge detection
            from scipy import ndimage
            sobel_x = ndimage.sobel(image, axis=1)
            sobel_y = ndimage.sobel(image, axis=0)
            gradient_magnitude = np.hypot(sobel_x, sobel_y)
            # Higher gradient variance indicates irregular borders
            border_score = min(1.0, np.std(gradient_magnitude) / 255.0)
            return border_score
    
    def _analyze_color(self, image: np.ndarray) -> float:
        """
        Analyze color variation (C).
        Score: 0-1, higher = more color variation
        """
        if len(image.shape) != 3:
            return 0.0
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Calculate color variance
            h_variance = np.var(hsv[:, :, 0])
            s_variance = np.var(hsv[:, :, 1])
            v_variance = np.var(hsv[:, :, 2])
            
            # Normalize
            color_variance = (h_variance + s_variance + v_variance) / 3
            max_variance = 255 ** 2
            
            color_score = min(1.0, color_variance / max_variance)
            
            return color_score
        except Exception:
            # Fallback: analyze color variation in RGB space
            r_variance = np.var(image[:, :, 0])
            g_variance = np.var(image[:, :, 1])
            b_variance = np.var(image[:, :, 2])
            
            color_variance = (r_variance + g_variance + b_variance) / 3
            max_variance = 255 ** 2
            
            color_score = min(1.0, color_variance / max_variance)
            return color_score
    
    def _analyze_diameter(self, clinical_features: Dict) -> float:
        """
        Analyze diameter (D).
        Score: 0-1, higher = larger diameter (>6mm is suspicious)
        """
        diameter_mm = clinical_features.get('diameter_mm', 0)
        
        if diameter_mm == 0:
            return 0.0
        
        # Score increases with diameter, especially >6mm
        if diameter_mm <= 6:
            score = diameter_mm / 6 * 0.5  # 0-0.5 for <6mm
        else:
            score = 0.5 + min(0.5, (diameter_mm - 6) / 10)  # 0.5-1.0 for >6mm
        
        return min(1.0, score)
    
    def _analyze_evolution(self, clinical_features: Dict) -> float:
        """
        Analyze evolution (E).
        Score: 0-1, higher = more concerning changes
        """
        evolution_data = clinical_features.get('evolution', {})
        
        if not evolution_data:
            return 0.0
        
        score = 0.0
        
        # Size change
        size_change = evolution_data.get('size_change_percent', 0)
        if size_change > 20:
            score += 0.4
        elif size_change > 10:
            score += 0.2
        
        # Color change
        color_change = evolution_data.get('color_change', False)
        if color_change:
            score += 0.3
        
        # Shape change
        shape_change = evolution_data.get('shape_change', False)
        if shape_change:
            score += 0.3
        
        return min(1.0, score)
    
    def _assess_risk(self, total_score: float) -> Dict:
        """Assess overall risk based on total score."""
        if total_score >= 0.7:
            risk_level = "High"
            risk_score = 85
            recommendations = [
                "High suspicion for melanoma",
                "Urgent dermatology referral",
                "Consider excisional biopsy",
                "Document with dermoscopy"
            ]
        elif total_score >= 0.5:
            risk_level = "Moderate-High"
            risk_score = 65
            recommendations = [
                "Moderate suspicion",
                "Dermatology evaluation recommended",
                "Consider biopsy",
                "Close follow-up"
            ]
        elif total_score >= 0.3:
            risk_level = "Moderate"
            risk_score = 45
            recommendations = [
                "Some concerning features",
                "Dermatology consultation",
                "Monitor closely",
                "Consider short-term follow-up"
            ]
        else:
            risk_level = "Low"
            risk_score = 25
            recommendations = [
                "Low suspicion",
                "Routine monitoring",
                "Patient education on self-examination"
            ]
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'recommendations': recommendations
        }
    
    def _interpret_scores(self, scores: Dict) -> str:
        """Provide interpretation of scores."""
        high_scores = [k for k, v in scores.items() if v > 0.6]
        
        if len(high_scores) >= 3:
            return "Multiple concerning features detected - high suspicion"
        elif len(high_scores) >= 2:
            return "Several concerning features - moderate suspicion"
        elif len(high_scores) == 1:
            return "One concerning feature - monitor closely"
        else:
            return "Low suspicion based on ABCDE criteria"

