from fastapi import status, APIRouter
from fastapi.responses import JSONResponse

from src.db import manager as mongo_manager

router = APIRouter(prefix="", tags=["status"])


@router.get('/health')
async def health():
    connected = await mongo_manager.connected()

    if not connected:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={'message': 'Service Unavailable'}
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content={'message': 'Ok'})
