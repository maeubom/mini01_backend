from fastapi import APIRouter, File, UploadFile
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import io

router = APIRouter()

# 모델과 피처 추출기 로드
model = ViTForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
feature_extractor = ViTFeatureExtractor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 파일 읽기
    image = Image.open(io.BytesIO(await file.read()))
    
    # 이미지 전처리
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1).item()  # 최상위 클래스 인덱스

    # 감정 클래스 매핑
    emotions = [
        "화남",
        "역겨움",
        "두려움",
        "기쁨",
        "슬픔",
        "놀람",
        "아무생각없음"
    ]

    return {"predicted_class": predictions, "emotion": emotions[predictions]}
