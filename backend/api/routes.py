from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
#from pydantic import BaseModel
from typing import List
from pathlib import Path
import uuid
import shutil

from backend.schemas.analysis import AnalysisResponse
from backend.services.agent_service import AgentService
from backend.services.evidence_service import create_evidence_preview

router = APIRouter()

"""
# pydantic request model
class AnalyzeRequest(BaseModel):
    query: str
    image_paths: list[str]
"""

## APIs for main backend using APIRouter
# 1. Health
@router.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "SatQuery AI",
        "version": "0.1.0",
    }

# 2. Analyze the user query -> Give result
# Getting the initiated router llm using Request
@router.post("/analyze")
def analyze(request: Request, query: str = Form(...), images: list[UploadFile] = File(...)):

    try:

        # 1. validate images
        if not images:
            raise HTTPException(
                status_code=400, detail="At least one TIFF image is required for analysis"
            )

        if len(images) > 2:
            raise HTTPException(
                status_code=400, detail="Maximum two TIFF images are allowed !"
            )

        # 2. validate image extension
        for image in images:

            extension = Path(image.filename).suffix.lower()

            if extension not in {".tif", ".tiff"}:
                raise HTTPException(
                    status_code=400, detail="Only .tif or .tiff images are allowed"
                )
            
        # 3. Request folder
        request_id = str(uuid.uuid4())

        upload_dir = Path("uploads") / request_id

        upload_dir.mkdir(parents=True, exist_ok=True)

        # 4. save files
        image_paths = []

        for index, image in enumerate(images):
            extension = Path(image.filename).suffix.lower()

            file_path = (upload_dir/f"Image_{index + 1}{extension}")

            with file_path.open("wb") as buffer:
                shutil.copyfileobj(
                    image.file,
                    buffer
                )

            image_paths.append(str(file_path))

        
        # 5. Create evidence previews
        evidence_images = []

        for image_path in image_paths:

            preview_path = create_evidence_preview(
                tiff_path=image_path,
                output_dir=str(upload_dir),
            )

            preview_filename = Path(preview_path).name

            evidence_images.append({
                "filename": preview_filename,
                "url": f"/api/v1/evidence/{request_id}/{preview_filename}",
            })


        # 6. Get agent service
        agent_service = request.app.state.agent_service


        # 7. Run LangGraph
        result = agent_service.analyze(
            query=query,
            image_paths=image_paths,
        )


        # 8. Return analysis + evidence
        return {
            "request_id": request_id,
            **result,
            "evidence": {
                "images": evidence_images,
            },
        }
    
    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))

# getting the image for preview at UI
@router.get("/evidence/{request_id}/{filename}")
def get_evidence(
    request_id: str,
    filename: str,
):
    file_path = (
        Path("uploads")
        / request_id
        / filename
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evidence image not found",
        )

    return FileResponse(
        file_path,
        media_type="image/jpeg",
    )
