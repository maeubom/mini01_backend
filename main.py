from fastapi import FastAPI, APIRouter
import uvicorn

# router 폴더 내 router 객체를 module1_router란 이름으로 alias하여 가져옴
from router.photo_senti import router as photo_router
from router.text_to_image import router as text_to_image_router
from router.text_senti import router as text_senti_router
from router.audio_to_text import router as transcribe_router


app = FastAPI()

# router를 app에 등록
app.include_router(photo_router)
app.include_router(text_to_image_router)
app.include_router(text_senti_router)
app.include_router(transcribe_router)

@app.get("/")
def index():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)