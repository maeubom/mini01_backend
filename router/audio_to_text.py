from fastapi import APIRouter, File, UploadFile
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pydub import AudioSegment
import librosa
import io

router = APIRouter()

# 모델 및 프로세서 로드
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

@router.post("/v1/api/audio-to-text")
async def transcribe(audio: UploadFile = File(...)):
    # 업로드된 오디오 파일을 읽음
    audio_bytes = await audio.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # webm을 wav로 변환
    audio = AudioSegment.from_file(audio_buffer, format="webm")
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    # librosa로 읽어들이기
    audio_data, rate = librosa.load(wav_buffer, sr=16000)

    # 전처리 및 입력 데이터 변환
    inputs = processor(audio_data, sampling_rate=rate, return_tensors="pt")

    # 추론 및 텍스트 생성
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {"text": transcription}