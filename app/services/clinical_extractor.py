import re
from typing import Dict, Any
from app.schemas import ExtractedPatientData

class ClinicalExtractor:
    """
    Parses raw OCR/extracted text to find clinical parameters.
    Does NOT invent or guess data. If it's not present, it's left as None.
    """
    
    def extract(self, text: str) -> ExtractedPatientData:
        data = ExtractedPatientData()
        confidence: Dict[str, float] = {}
        data.raw_text = text.strip()
        
        # Simple extraction using regex for prototype
        
        # Age: e.g., "Age: 58", "58 y.o."
        age_match = re.search(r'(?:Age[\s:]+|)(\d{1,3})\s*(?:y\.?o\.?|years old|Y/M|yrs)', text, re.IGNORECASE)
        if not age_match:
             age_match = re.search(r'Age[\s:]+(\d{1,3})', text, re.IGNORECASE)
        if age_match:
            data.age = int(age_match.group(1))
            confidence['age'] = 0.90
            
        # Gender: e.g., "Sex: Male", "48 y.o. male"
        gender_match = re.search(r'\b(male|female)\b', text, re.IGNORECASE)
        if gender_match:
            g = gender_match.group(1).lower()
            data.gender = "male" if g == "male" else "female"
            confidence['gender'] = 0.90
            
        # BP: e.g., "BP: 165/100", "Blood Pressure 165 / 100"
        bp_match = re.search(r'(?:BP|Blood Pressure)[\s:]*(\d{2,3})\s*/\s*(\d{2,3})', text, re.IGNORECASE)
        if bp_match:
            data.systolic_bp = float(bp_match.group(1))
            data.diastolic_bp = float(bp_match.group(2))
            confidence['systolic_bp'] = 0.85
            confidence['diastolic_bp'] = 0.85

        # Heart rate / Pulse: e.g., "Pulse: 105", "HR: 80", "Pulse Rate 74"
        hr_match = re.search(r'(?:Pulse(?: Rate)?|HR|Heart Rate)[\s:]*(\d{2,3})', text, re.IGNORECASE)
        if hr_match:
            data.heart_rate = float(hr_match.group(1))
            confidence['heart_rate'] = 0.90
            
        # Blood Sugar: e.g., "Blood Sugar: 220"
        bs_match = re.search(r'(?:Blood Sugar|Blood Glucose)[\s:]*(\d{2,3})', text, re.IGNORECASE)
        if bs_match:
            data.blood_glucose = float(bs_match.group(1))
            confidence['blood_glucose'] = 0.90
            
        # Chronic Diseases
        diseases = []
        if re.search(r'diabet', text, re.IGNORECASE):
            diseases.append("diabetes")
        if re.search(r'hypertension|HTN', text, re.IGNORECASE):
            diseases.append("hypertension")
        if re.search(r'asthma', text, re.IGNORECASE):
            diseases.append("asthma")
            
        if diseases:
            data.chronic_diseases = diseases
            confidence['chronic_diseases'] = 0.80
            
        # Missing followups
        miss_match = re.search(r'Missed[\w\s]*appointments?[\s:]*(\d+)', text, re.IGNORECASE)
        if miss_match:
            data.missed_followups = int(miss_match.group(1))
            confidence['missed_followups'] = 0.80
            
        # Patient Name (Simple assumption: "Patient: XYZ")
        name_match = re.search(r'Patient(?: Name)?[\s:]+([A-Za-z\s]+)(?:\n|$)', text, re.IGNORECASE)
        if name_match:
            data.patient_name = name_match.group(1).strip()
            confidence['patient_name'] = 0.80
            
        data.extraction_confidence = confidence
        return data
