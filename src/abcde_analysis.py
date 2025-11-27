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
import cv2


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
        if len(lesion_image.shape) == 3:
            gray_image = cv2.cvtColor(lesion_image, cv2.COLOR_RGB2GRAY)
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
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
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
    
    def _analyze_color(self, image: np.ndarray) -> float:
        """
        Analyze color variation (C).
        Score: 0-1, higher = more color variation
        """
        if len(image.shape) != 3:
            return 0.0
        
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

