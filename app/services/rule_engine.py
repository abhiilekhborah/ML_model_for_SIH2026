from typing import Tuple, List
from app.schemas import ExtractedPatientData

class SafetyRuleEngine:
    """
    Applies deterministic clinical rules that override ML predictions if necessary.
    """
    
    def evaluate(self, data: ExtractedPatientData, ml_prediction: str) -> Tuple[str, bool, List[str]]:
        """
        Returns: final_risk_level (str), rule_override (bool), rule_reasons (List[str])
        """
        override = False
        reasons = []
        current_risk = ml_prediction
        
        # Rule 1: Hypertensive Crisis
        if data.systolic_bp is not None and data.systolic_bp >= 180:
            if current_risk != "HIGH":
                current_risk = "HIGH"
                override = True
            reasons.append("A configured safety rule escalated the final risk level due to critically high systolic blood pressure (>=180).")
            
        # Rule 2: Critical Blood Sugar
        if data.blood_glucose is not None and data.blood_glucose >= 300:
            if current_risk != "HIGH":
                current_risk = "HIGH"
                override = True
            reasons.append("A configured safety rule escalated the final risk level due to critically high blood glucose (>=300).")
            
        # Rule 3: High Risk Pregnancy
        if data.high_risk_pregnancy:
             if current_risk != "HIGH":
                current_risk = "HIGH"
                override = True
             reasons.append("A configured safety rule escalated the final risk level due to high risk pregnancy.")

        return current_risk, override, reasons
