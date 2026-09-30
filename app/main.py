from fastapi import FastAPI
from .models import SearchRequest, SearchResponse, AskRequest, AskResponse
from .services import search, ask

app = FastAPI(
    title="Daily Planet RAG",
    description="Retrieval-augmented generation service for the Daily Planet",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    results = search(req.query, req.k)
    return SearchResponse(results = results)

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    result = ask(req.question)
    return AskResponse(**result)