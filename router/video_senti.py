import tempfile
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException
import torch
import cv2
from PIL import Image
import numpy as np
import os
from collections import Counter

app = FastAPI()
router = APIRouter()

# 사진으로 감정분석하는 모델입니다.
video_cls = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition", use_fast=True)

@router.post("/video/")
async def create_upload_file(file: UploadFile = File(...)):
    # 파일 형식 체크
    if not (file.content_type == 'video/mp4' or file.content_type == 'video/webm'):
        raise HTTPException(status_code=400, detail="사용불가능한 확장자입니다. 'mp4'와 'webm' 형식만 가능합니다.")
    
    # 파일 불러오기
    video_stream = await file.read()
    
    # 임시파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_stream)
        tmp_file_path = tmp_file.name

    # OpenCV로 비디오 파일을 읽어 프레임을 추출
    cap = cv2.VideoCapture(tmp_file_path)
    
    emotions = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 매 30프레임마다 처리
        if frame_count % 30 == 0:
            # 프레임을 BGR에서 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OpenCV 이미지에서 PIL 이미지로 변환
            pil_image = Image.fromarray(rgb_frame)

            # Run inference
            with torch.no_grad():
                outputs = video_cls(pil_image)
            
            predicted_emotion = outputs[0]['label']
            emotions.append(predicted_emotion)

        frame_count += 1

    cap.release()
    os.remove(tmp_file_path)
    
    # 가장 자주 나오는 감정 찾기
    most_common_emotion = Counter(emotions).most_common(1)
    most_common_emotion_label = most_common_emotion[0][0] if most_common_emotion else None

    return {
        "filename": file.filename,
        "emotions": emotions,
        "most_common_emotion": most_common_emotion_label
    }

app.include_router(router)
