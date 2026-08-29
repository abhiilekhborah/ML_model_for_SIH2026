from fastapi import FastAPI, UploadFile, File
from app.schemas import ExtractedPatientData, PredictionResponse
from app.services.document_extractor import DocumentExtractor
from app.services.risk_aggregator import RiskAggregator
from datetime import datetime, timezone
import logging

app = FastAPI(title="MediQuick ML Risk Prediction Service")

document_extractor = DocumentExtractor()
risk_aggregator = RiskAggregator()

@app.post("/predict/document", response_model=PredictionResponse)
async def predict_from_document(file: UploadFile = File(...)):
    """
    Process an uploaded prescription or medical report and return a risk assessment.
    Accepts JPG, PNG, PDF.
    """
    try:
        extracted_data = await document_extractor.extract(file)
    except Exception as e:
        # Document Extraction Failed
        return PredictionResponse(
            risk_level="UNKNOWN",
            risk_score=0.0,
            confidence=0.0,
            ml_prediction="UNKNOWN",
            rule_override=False,
            reasons=[],
            data_completeness=0.0,
            model_version="1.0.0",
            assessed_at=datetime.now(timezone.utc).isoformat(),
            error="DOCUMENT_EXTRACTION_FAILED",
            message=str(e)
        )
        
    return risk_aggregator.process(extracted_data)

@app.post("/predict", response_model=PredictionResponse)
async def predict_structured(data: ExtractedPatientData):
    """
    Process structured patient data directly (bypasses OCR/Extraction).
    Uses the exact same risk aggregation pipeline.
    """
    return risk_aggregator.process(data)
