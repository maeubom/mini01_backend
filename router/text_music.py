from fastapi import FastAPI, Form, APIRouter, HTTPException
from fastapi.responses import Response
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch
import io
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from gridfs import GridFS
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Union

# .env 파일 로드
load_dotenv()

# MongoDB 연결 설정
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["maeubom"]
fs = GridFS(db)
generated_music_col = db["GeneratedMusic"]

# FastAPI 인스턴스 및 라우터 설정
app = FastAPI()
router = APIRouter()

# MusicGen 모델 설정
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torch_dtype=torch.float32).to("cuda")

def generate_music_binary(text: str, length: int = 512) -> Dict[str, Union[bytes, int]]:
    """
    텍스트를 바탕으로 음악을 생성하고 바이너리 데이터로 반환
    """
    try:
        # 텍스트를 모델에 입력할 수 있는 형태로 변환
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
        
        # 모델을 통해 오디오 값 생성
        audio_values = model.generate(**inputs, max_new_tokens=length)
        
        # WAV 파일 데이터를 메모리에 직접 생성
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        
        # WAV 파일을 바이너리 데이터로 변환
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_data)
        buffer.seek(0)
        
        return {
            "audio_binary": buffer.getvalue(),
            "sampling_rate": sampling_rate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음악 생성 실패: {str(e)}")


# POST 메서드: 음악 생성 및 GridFS에 저장
@app.post("/create_music/binary/")
async def create_music_binary_endpoint(summary_text: str = Form(...), length: int = Form(512)):
    # 음악 생성
    result = generate_music_binary(summary_text, length)
    
    # GridFS에 음악 바이너리 데이터 저장
    file_id = fs.put(result["audio_binary"], filename="generated_music.wav")
    
    # MongoDB에 메타데이터 저장 (file_id 포함)
    music_data = {
        "file_name": "generated_music.wav",
        "file_id": file_id,
        "summary_text": summary_text,
        "length": length,
        "sampling_rate": result["sampling_rate"],
        "created_at": datetime.utcnow()
    }
    generated_music_col.insert_one(music_data)
    
    # 생성된 파일 정보 반환
    return {"file_name": "generated_music.wav", "file_id": str(file_id)}


# GET 메서드: GridFS에서 특정 노래 파일을 바이너리로 반환
@app.get("/create_music/binary")
async def get_create_music(file_id: str):
    # MongoDB에서 파일 정보 검색
    music_data = generated_music_col.find_one({"file_id": file_id})
    
    if not music_data:
        raise HTTPException(status_code=404, detail="Music file not found in database")
    
    # GridFS에서 파일 데이터를 읽어 반환
    try:
        audio_data = fs.get(file_id).read()
    except Exception:
        raise HTTPException(status_code=404, detail="Music file not found in GridFS")
    
    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={music_data['file_name']}"
        }
    )

# 라우터 추가
app.include_router(router)
