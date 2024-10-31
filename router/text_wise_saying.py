from transformers import pipeline, OPTForCausalLM, GPT2Tokenizer
from fastapi import FastAPI, Form, APIRouter
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import torch
import random
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MongoDB 연결 설정
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["maeubom"]
quote_col = db["Quote"]

# 모델과 토크나이저 설정
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

# 모델이 사용할 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 명언 파일에서 명언을 읽어 리스트에 저장
def load_quotes(file_path="../wise_saying.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

quotes = load_quotes()  # wise_saying.txt에서 명언 불러오기

# FastAPI 인스턴스 및 라우터 설정
app = FastAPI()
router = APIRouter()

# FastAPI 엔드포인트: 입력한 텍스트를 기반으로 명언 생성
@app.post("/create_text/")
async def create_text(input_text: str = Form(...)):
    # 명언 리스트에서 무작위로 하나 선택
    selected_quote = random.choice(quotes)
    selected_quote = selected_quote.split('.', 1)[-1].strip()

    # MongoDB에 명언 저장
    quote_data = {
        "quote": selected_quote,
        "input_text": input_text,
        "created_at": datetime.utcnow()
    }
    quote_col.insert_one(quote_data)
    
    return {"quote": selected_quote, "message": "Quote saved to database"}

# 명언 가져오기
@app.get("/create_text/")
async def get_create_text():
    # MongoDB에서 저장된 명언 중 무작위로 하나 가져오기
    count = quote_col.count_documents({})
    if count == 0:
        return {"message": "No quotes found in the database"}

    random_index = random.randint(0, count - 1)
    random_quote = quote_col.find().skip(random_index).limit(1)[0]

    return {
        "quote": random_quote["quote"],
        "input_text": random_quote["input_text"],
        "created_at": random_quote["created_at"]
    }

# 라우터 추가
app.include_router(router)
