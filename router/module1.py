from fastapi import APIRouter


router = APIRouter()


@router.get("/module1")
def module1():
    return {"message": "Hi, im module1"}