from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExtractedPatientData(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    temperature: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    blood_glucose: Optional[float] = None
    chronic_diseases: List[str] = Field(default_factory=list)
    previous_hospitalizations: Optional[int] = None
    previous_referrals: Optional[int] = None
    missed_followups: Optional[int] = None
    medication_adherence: Optional[float] = None
    pregnancy: Optional[bool] = None
    high_risk_pregnancy: Optional[bool] = None
    medications: List[str] = Field(default_factory=list)
    
    # Internal metadata
    extraction_confidence: Dict[str, float] = Field(default_factory=dict)
    raw_text: Optional[str] = None
    
class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: float
    confidence: float
    ml_prediction: str
    rule_override: bool
    reasons: List[str]
    data_completeness: float
    extracted_data: Optional[ExtractedPatientData] = None
    extraction_warnings: List[str] = Field(default_factory=list)
    model_version: str
    assessed_at: str
    error: Optional[str] = None
    message: Optional[str] = None
