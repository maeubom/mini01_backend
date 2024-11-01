from fastapi import FastAPI, Form, HTTPException, APIRouter
from fastapi.responses import Response
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch
import io

# FastAPI 인스턴스 설정

app = FastAPI()
router = APIRouter()

# MusicGen 모델 설정
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torch_dtype=torch.float32).to("cuda")

def generate_music_binary(text: str, length: int = 512):
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

# POST 메서드: 음악 생성 및 바이너리로 반환
@router.post("/v1/api/text-to-music")
async def create_music_binary_endpoint(summary_text: str = Form(...), length: int = Form(512)):
    # 음악 생성
    result = generate_music_binary(summary_text, length)
    
    # 생성된 음악 바이너리 데이터 반환
    return Response(
        content=result["audio_binary"],
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_music.wav"
        }
    )
