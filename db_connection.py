from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
from datetime import datetime


# .env 파일 로드
load_dotenv()

uri = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# 데이터베이스 및 컬렉션 선택
db = client["maeubom"]
collection = db["maeubom"]

# 삽입할 데이터의 속성 정의
# 각 컬렉션 정의
emotion_analysis_col = db["EmotionAnalysis"]
summary_col = db["Summary"]
generated_image_col = db["GeneratedImage"]
generated_music_col = db["GeneratedMusic"]
quote_col = db["Quote"]
analysis_session_col = db["AnalysisSession"]

def get_database():
    return db

'''
# 다른 파일에서 db 연결하기
from database import get_database

db = get_database()  # 데이터베이스 객체 가져오기
collection = db["maeumbom"]  # 컬렉션 선택
'''