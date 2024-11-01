from transformers import pipeline
from fastapi import FastAPI, Form, APIRouter

# 요약 파이프라인 설정
summarize_model = pipeline("summarization", model="lcw99/t5-base-korean-text-summary", device=0)

# FastAPI 인스턴스 설정
app = FastAPI()
router = APIRouter()

# 요약 함수 정의
def text_sum(text: str) -> str:
    """
    입력 텍스트를 요약하여 반환합니다.
    """
    summary = summarize_model(text, max_length=100, min_length=10, do_sample=False)
    return summary[0]["summary_text"]

# API 엔드포인트 정의
@app.post("/text_sum/")
async def text_summary(input_text: str = Form(...)):
    predicted_summary = text_sum(input_text)
    return {"result": predicted_summary}
