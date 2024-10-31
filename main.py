from fastapi import FastAPI, APIRouter
import uvicorn

# router 폴더 내 router 객체를 module1_router란 이름으로 alias하여 가져옴
from router.text_senti import router as text_senti_router


app = FastAPI()

# router를 app에 등록
app.include_router(text_senti_router)


@app.get("/")
def index():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)