from fastapi import FastAPI, Form, APIRouter, HTTPException
from fastapi.responses import Response
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch
import io
import base64
from typing import Optional, Dict, Union

# FastAPI 인스턴스 및 라우터 설정
app = FastAPI()
router = APIRouter()

# # MusicGen 모델 설정
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


# FastAPI 엔드포인트: 바이너리 데이터로 직접 반환
@app.post("/recommend_music/binary")
async def recommend_music_binary(summary_text: str = Form(...), length: int = Form(512)):
    result = generate_music_binary(summary_text, length)
    
    return Response(
        content=result["audio_binary"],
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_music.wav"
        }
    )


# 라우터 추가
app.include_router(router)
