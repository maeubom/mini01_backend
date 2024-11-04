from transformers import pipeline, OPTForCausalLM, GPT2Tokenizer
from fastapi import FastAPI, Form, APIRouter
from datetime import datetime
import torch
import random


# 모델과 토크나이저 설정
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

# 모델이 사용할 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 명언 파일에서 명언을 읽어 리스트에 저장
def load_quotes(file_path="wise_saying.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

quotes = load_quotes()  # wise_saying.txt에서 명언 불러오기

# FastAPI 인스턴스 설정

app = FastAPI()
router = APIRouter()

# FastAPI 엔드포인트: 입력한 텍스트를 기반으로 명언 생성
@router.post("/v1/api/text-to-wise-saying")
async def create_text(input_text: str = Form(...)):
    # 명언 리스트에서 무작위로 하나 선택
    selected_quote = random.choice(quotes)
    selected_quote = selected_quote.split('.', 1)[-1].strip()
    
    # 생성된 명언 반환
    return {
        "quote": selected_quote,
        "input_text": input_text,
        "created_at": datetime.now()
    }