from transformers import pipeline
from fastapi import FastAPI, Form, APIRouter, HTTPException
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import nltk
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MongoDB 연결 설정
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["maeubom"]
summary_col = db["Summary"]

# NLTK 데이터 다운로드
nltk.download('punkt')

# STEP 1: 요약 파이프라인 설정
summarize_model = pipeline("summarization", model="lcw99/t5-base-korean-text-summary", device=0)

# FastAPI 인스턴스 및 라우터 설정
app = FastAPI()
router = APIRouter()

# STEP 2: 요약 함수 정의
def text_sum(text: str) -> str:
    """
    입력 텍스트를 요약하여 반환합니다.
    """
    summary = summarize_model(text, max_length=100, min_length=10, do_sample=False)
    return summary[0]["summary_text"]

# STEP 3: API 엔드포인트 정의
@router.post("/text_sum/")
async def text_summary(input_text: str = Form(...)):
    predicted_summary = text_sum(input_text)
    
    # MongoDB에 요약 데이터 저장
    summary_data = {
        "summary": predicted_summary
    }
    summary_col.insert_one(summary_data)

    return {"result": predicted_summary}

# STEP 4: GET API 엔드포인트 정의 (저장된 요약 데이터 가져오기)
@router.get("/text_sum/")
async def get_text_summaries():
    summaries = list(summary_col.find({}, {"_id": 0, "summary": 1}))
    if not summaries:
        raise HTTPException(status_code=404, detail="No summaries found in the database")

    return {"summaries": summaries}


# 라우터 추가
app.include_router(router)
