# mini01_backend
마음:봄 미니프로젝트 backend 관리 저장소 입니다.

사진을 올리면 감정분석 해주는 모델입니다. </br>
감정은 총 6가지로 인덱스 0~5번까지 분류되고, </br>
"화남", "역겨움", "두려움", "기쁨", "슬픔", "놀람", "아무생각없음"</br>
이렇게 분류 되어 출력됩니다.
</br></br>
## Text To Image
영어로 감정 쿼리를 넣으면 귀여운 고양이(.png)가 나옵니다.

서버 IP는 임시 IP이니 주의 바랍니다.

</br></br>

## 필요 라이브러리 명령어 </br>
# pip install librosa scipy websocket-client fastapi uvicorn torch transformers opencv-python Pillow numpy </br>
이 명령어면 모든 모델의 필수 라이브러리가 전부 설치됩니다.
