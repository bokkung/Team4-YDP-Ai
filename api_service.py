import logging
import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# 1. โหลดตัวแปรจาก .env (บรรทัดนี้สำคัญมาก มันจะไปดึง MERCIL_API_KEY มา)
load_dotenv()

# Import Core Logic
from search_pipeline import (
    execute_search, 
    get_chroma_collection, 
    get_embedding_model, 
    EMB_MODEL_NAME, 
    VECTOR_DB_PATH, 
    COLLECTION_NAME, 
    logger
)

# ==========================================
# 🔐 SECURITY ZONE: ตรวจสอบกุญแจ (Bearer Token)
# ==========================================
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    ฟังก์ชันนี้จะถูกเรียกทุกครั้งที่มีคนยิง API เข้ามา
    """
    token = credentials.credentials
    
    # ดึงค่า Key ที่คุณตั้งไว้ใน .env
    REAL_API_KEY = os.getenv("MERCIL_API_KEY") 
    
    # ถ้าลืมตั้งใน .env จะแจ้งเตือน (แต่ไม่ error)
    if not REAL_API_KEY:
        logger.warning("⚠️ Warning: MERCIL_API_KEY not set in .env. API is unsecured!")
        return token

    # เช็คว่า Key ที่ส่งมา ตรงกับใน .env ไหม
    if token != REAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authentication (Wrong API Key)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
# ==========================================


# --- Data Models ---
class SearchRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = {} 

class SearchResponse(BaseModel):
    query: str
    intent_detected: Dict[str, Any]
    results: List[Dict[str, Any]]
    
# --- App Init ---
app = FastAPI(
    title="Mercil AI API",
    description="AI Real Estate Search (Secured with Bearer Token)",
    version="1.0.0"
)

# --- Startup ---
@app.on_event("startup")
def startup_event():
    try:
        logger.info("Loading Embedding Model and ChromaDB Collection...")
        app.state.embed_model = get_embedding_model(EMB_MODEL_NAME)
        app.state.collection = get_chroma_collection(VECTOR_DB_PATH, COLLECTION_NAME)
        logger.info("Startup complete. Service is ready.")
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        raise RuntimeError("Service initialization failed.")

# --- API Endpoint (LOCKED 🔒) ---
@app.post("/api/v1/search", 
          response_model=SearchResponse, 
          tags=["Search"],
          dependencies=[Depends(verify_api_key)]) # <--- บรรทัดนี้คือแม่กุญแจ
async def search_endpoint(request: SearchRequest):
    try:
        logger.info(f"Received query: '{request.query}'")
        
        search_output = execute_search(
            query=request.query, 
            filters=request.filters,
            embed_model=app.state.embed_model,
            collection=app.state.collection
        )
        return search_output
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Pipeline Error")

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)