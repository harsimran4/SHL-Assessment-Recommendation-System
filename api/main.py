from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from recommender import get_recommendations

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class QueryRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: QueryRequest):
    results = get_recommendations(req.query)
    return {"recommended_assessments": results}