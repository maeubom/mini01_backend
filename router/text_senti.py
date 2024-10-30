# STEP 1
from transformers import pipeline

# STEP 2

## get_bi_sentiment Model
bi_sentiment_model = pipeline(model="Copycats/koelectra-base-v3-generalized-sentiment-analysis", device=0)
# sentiment_model = pipeline(model="WhitePeak/bert-base-cased-Korean-sentiment", device=0)

## get_top_k_sentiment Model
top_k_sentiment_model = pipeline(model="hun3359/klue-bert-base-sentiment", device=0)



# 0에 가까울수록 부정적. 1에 가까울수록 긍정적.
def get_bi_sentiment(text: str):
    result = bi_sentiment_model(text, top_k=None)
    filtered_result = [res for res in result if res['label'] == '1']
    return filtered_result[0]


def get_top_k_sentiment(text: str, top_k: int):
    result = top_k_sentiment_model(text, top_k=top_k)
    return result
## -------------------------------------------------------------------------------

from fastapi import APIRouter, Form


router = APIRouter()


@router.post("/senti-2")
def senti2(query: str = Form(...)):
    result = get_bi_sentiment(query)
    return result


@router.post("/senti-k")
def sentik(query: str = Form(...), top_k: int = Form(...)):
    result = get_top_k_sentiment(query, top_k)
    return result