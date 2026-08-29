from datetime import datetime, timezone
from app.schemas import ExtractedPatientData, PredictionResponse
from app.services.ml_pipeline import MLPipeline
from app.services.rule_engine import SafetyRuleEngine

class RiskAggregator:
    """
    Orchestrates the ML Pipeline and Rule Engine to produce the final Response.
    """
    
    def __init__(self):
        self.ml_pipeline = MLPipeline()
        self.rule_engine = SafetyRuleEngine()
        
    def process(self, data: ExtractedPatientData) -> PredictionResponse:
        try:
            ml_prediction, risk_score, data_completeness, _ = self.ml_pipeline.predict(data)
        except ValueError as e:
            if str(e) == "INSUFFICIENT_CLINICAL_DATA":
                return PredictionResponse(
                    risk_level="UNKNOWN",
                    risk_score=0.0,
                    confidence=0.0,
                    ml_prediction="UNKNOWN",
                    rule_override=False,
                    reasons=[],
                    data_completeness=0.0,
                    extracted_data=data,
                    model_version="1.0.0",
                    assessed_at=datetime.now(timezone.utc).isoformat(),
                    error="INSUFFICIENT_CLINICAL_DATA",
                    message="The document did not contain enough usable clinical information to generate a reliable risk assessment."
                )
            raise e
            
        final_risk_level, rule_override, rule_reasons = self.rule_engine.evaluate(data, ml_prediction)
        
        # Build Reasons List
        reasons = []
        if data.chronic_diseases:
            reasons.append("Existing chronic conditions contributed to the risk assessment.")
        if data.systolic_bp is not None or data.diastolic_bp is not None:
            reasons.append("Recorded blood pressure contributed to the risk assessment.")
            
        reasons.extend(rule_reasons)
        
        if not reasons:
             reasons.append("Risk assessed based on baseline patient profile.")
             
        # Extraction warnings (Confidence checking)
        warnings = []
        if data.extraction_confidence:
             for field, conf in data.extraction_confidence.items():
                 if conf < 0.85:
                     warnings.append(f"Low extraction confidence for field '{field}'.")
                     
        if warnings:
            warnings.append("Some clinical information could not be reliably extracted from the uploaded document.")
            
        return PredictionResponse(
            risk_level=final_risk_level,
            risk_score=risk_score,
            confidence=risk_score if not rule_override else 1.0, # Rules are 100% confident
            ml_prediction=ml_prediction,
            rule_override=rule_override,
            reasons=reasons,
            data_completeness=data_completeness,
            extracted_data=data,
            extraction_warnings=warnings,
            model_version="1.0.0",
            assessed_at=datetime.now(timezone.utc).isoformat()
        )
