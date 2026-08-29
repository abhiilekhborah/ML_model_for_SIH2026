from typing import Dict, Any, Tuple
from app.schemas import ExtractedPatientData
import random

class MLPipeline:
    """
    Simulated ML Risk Prediction Model.
    In a real system, this would use a scikit-learn or XGBoost model.
    """
    
    def predict(self, data: ExtractedPatientData) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Returns: ml_prediction (str), risk_score (float), data_completeness (float), imputed_features (dict)
        """
        # 1. Feature Engineering & Missing Data Handling
        imputed_features = self._preprocess(data)
        
        # Calculate data completeness
        total_fields = 9 # Core fields we expect
        present_fields = sum(1 for k, v in imputed_features.items() if getattr(data, k, None) is not None)
        data_completeness = present_fields / total_fields
        
        if data_completeness < 0.3:
            raise ValueError("INSUFFICIENT_CLINICAL_DATA")

        # 2. ML Prediction (Simulated Logic)
        score = 0.1
        
        if imputed_features['age'] > 65:
            score += 0.2
        if imputed_features['systolic_bp'] > 140:
            score += 0.2
        if imputed_features['heart_rate'] > 100:
            score += 0.15
        if imputed_features['blood_glucose'] > 140:
            score += 0.2
            
        if "diabetes" in data.chronic_diseases or "hypertension" in data.chronic_diseases:
            score += 0.2
            
        # Add a tiny bit of random noise for realism
        score += random.uniform(-0.05, 0.05)
        score = min(max(score, 0.0), 1.0)
        
        # Classification
        if score > 0.7:
            prediction = "HIGH"
        elif score > 0.4:
            prediction = "MEDIUM"
        else:
            prediction = "LOW"
            
        return prediction, score, data_completeness, imputed_features
        
    def _preprocess(self, data: ExtractedPatientData) -> Dict[str, Any]:
        """
        Internal Imputation. We DO NOT modify the original ExtractedPatientData.
        We only create a feature dict for the ML model.
        """
        features = {
            'age': data.age if data.age is not None else 45,
            'gender': data.gender if data.gender is not None else 'unknown',
            'systolic_bp': data.systolic_bp if data.systolic_bp is not None else 120.0,
            'diastolic_bp': data.diastolic_bp if data.diastolic_bp is not None else 80.0,
            'heart_rate': data.heart_rate if data.heart_rate is not None else 72.0,
            'temperature': data.temperature if data.temperature is not None else 37.0,
            'oxygen_saturation': data.oxygen_saturation if data.oxygen_saturation is not None else 98.0,
            'blood_glucose': data.blood_glucose if data.blood_glucose is not None else 100.0,
            'missed_followups': data.missed_followups if data.missed_followups is not None else 0
        }
        return features
