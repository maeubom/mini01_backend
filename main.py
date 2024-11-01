from fastapi import FastAPI, APIRouter
import uvicorn

from router.video_senti import router as video_router
from router.text_to_image import router as text_to_image_router
from router.text_senti import router as text_senti_router
from router.audio_to_text import router as transcribe_router
from router.text_sum import router as text_sum_router
from router.text_music import router as text_music_router
from router.text_to_image import router as text_image_router
from router.text_wise_saying import router as text_wise_saying_router

app = FastAPI()
router = APIRouter()

# router를 app에 등록
app.include_router(video_router)
app.include_router(text_to_image_router)
app.include_router(text_senti_router)
app.include_router(transcribe_router)
app.include_router(text_sum_router)
app.include_router(text_music_router)
app.include_router(text_image_router)
app.include_router(text_wise_saying_router)

@app.get("/")
def index():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)