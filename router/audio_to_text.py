from fastapi import FastAPI, File, UploadFile
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import io

app = FastAPI()

# 모델 및 프로세서 로드
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 업로드된 오디오 파일을 읽음
    audio_bytes = await audio.read()
    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, rate = librosa.load(audio_buffer, sr=16000)

    # 전처리 및 입력 데이터 변환
    inputs = processor(audio_data, sampling_rate=rate, return_tensors="pt")


    # 추론 및 텍스트 생성
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {"text": transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
