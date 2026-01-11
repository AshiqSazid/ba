from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any
import structlog

from app.core.database import get_db
from app.schemas.therapy import (
    ExportRequest as NewExportRequest, ExportResponse as NewExportResponse,
    TherapyRecommendation, ExportRecommendation
)
from app.services.export_service import ExportService

logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize export service
export_service = ExportService()


@router.post("/export", response_model=NewExportResponse)
async def export_data(
    request: NewExportRequest
):
    """
    Export patient data and recommendations in various formats.

    Supported formats:
    - pdf: Generate a formatted PDF report with patient information, recommendations, and analysis
    - docx: Generate a Microsoft Word document report
    - csv: Export raw data in CSV format for analysis
    - json: Export structured data in JSON format

    The export includes:
    - Patient demographic and clinical information
    - Big Five personality scores (if available)
    - Music recommendations with metadata and scores
    - Session summary and therapeutic insights
    - Algorithm information and timestamps

    The response contains a base64-encoded file that can be downloaded by the client.
    """
    try:
        logger.info(
            "Received export request",
            export_format=request.format,
            num_recommendations=len(request.recommendations),
            has_patient_info=bool(request.patient_info),
            has_big_five=bool(request.big_five)
        )

        # Validate export format
        supported_formats = ['pdf', 'docx', 'csv', 'json']
        if request.format not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported formats: {', '.join(supported_formats)}"
            )

        # Validate required data
        if not request.patient_info:
            raise HTTPException(status_code=400, detail="Patient information is required")

        if not request.recommendations:
            raise HTTPException(status_code=400, detail="At least one recommendation is required")

        # Convert recommendation dicts to objects for export service
        from app.schemas.therapy import TherapyRecommendation

        def _convert_to_therapy_rec(raw_rec: Any, fallback_id: int) -> TherapyRecommendation:
            """
            Normalize inbound recommendation payloads so the export service
            always works with TherapyRecommendation objects.
            """
            # Extract dict data from possible Pydantic models
            if hasattr(raw_rec, "model_dump"):
                rec_data = raw_rec.model_dump()
            elif hasattr(raw_rec, "dict"):
                rec_data = raw_rec.dict()
            elif isinstance(raw_rec, dict):
                rec_data = raw_rec
            else:
                raise HTTPException(status_code=400, detail="Invalid recommendation payload")

            if not rec_data.get('id'):
                rec_data['id'] = fallback_id

            if not rec_data.get('session_id'):
                rec_data['session_id'] = (request.patient_summary or {}).get('session_id')

            if not rec_data.get('session_id'):
                raise HTTPException(status_code=400, detail="Each recommendation must include a session_id")

            # Sanitize year to avoid validation errors - handle historical music
            if 'year' in rec_data and rec_data['year'] is not None:
                if rec_data['year'] < 1500:
                    rec_data['year'] = None  # Remove clearly invalid years
                elif rec_data['year'] > 2100:
                    rec_data['year'] = 2023  # Cap future years to current year

            created_at_value = rec_data.get('created_at')
            if isinstance(created_at_value, str) and created_at_value:
                try:
                    rec_data['created_at'] = datetime.fromisoformat(created_at_value)
                except ValueError:
                    rec_data['created_at'] = datetime.utcnow()
            elif created_at_value is None:
                rec_data['created_at'] = datetime.utcnow()

            return TherapyRecommendation(**rec_data)

        therapy_recommendations = [
            _convert_to_therapy_rec(rec, idx)
            for idx, rec in enumerate(request.recommendations, start=1)
        ]

        # Generate export
        export_result = export_service.export_data(
            export_format=request.format,
            patient_info=request.patient_info,
            recommendations=therapy_recommendations,
            big_five=request.big_five,
            patient_summary=request.patient_summary
        )

        logger.info(
            "Successfully exported data",
            format=request.format,
            filename=export_result['filename'],
            file_size=export_result['file_size']
        )

        return NewExportResponse(**export_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export/formats")
async def get_export_formats():
    """
    Get list of supported export formats with descriptions.
    """
    return {
        "formats": [
            {
                "id": "pdf",
                "name": "PDF Report",
                "description": "Formatted PDF report suitable for printing and clinical documentation",
                "content_type": "application/pdf",
                "features": ["Professional formatting", "Patient summary", "Recommendation details", "Therapeutic insights"]
            },
            {
                "id": "docx",
                "name": "Word Document",
                "description": "Microsoft Word document for editing and collaboration",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "features": ["Editable format", "Structured content", "Tables for data", "Easy customization"]
            },
            {
                "id": "csv",
                "name": "CSV Data",
                "description": "Comma-separated values for data analysis and import",
                "content_type": "text/csv",
                "features": ["Raw data export", "Spreadsheet compatible", "Analysis ready", "Machine learning format"]
            },
            {
                "id": "json",
                "name": "JSON Data",
                "description": "Structured JSON data for programmatic access",
                "content_type": "application/json",
                "features": ["Structured data", "API compatible", "Full metadata", "Machine readable"]
            }
        ]
    }


@router.get("/export/health")
async def health_check():
    """
    Health check endpoint for the export service.
    """
    return {
        "status": "healthy",
        "service": "export",
        "supported_formats": ["pdf", "docx", "csv", "json"],
        "export_directory": str(export_service.export_dir)
    }
