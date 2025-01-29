from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from generator import SmolLM2Generator
import uvicorn
from typing import Optional, List
from datetime import datetime
import collections

app = FastAPI(title="SmolLM2 API")
templates = Jinja2Templates(directory="templates")

# Create a deque to store the last N requests
MAX_LOGS = 10
request_logs = collections.deque(maxlen=MAX_LOGS)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.5

generator = SmolLM2Generator()

@app.get("/")
async def home(request: Request):
    """Render the monitoring interface"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "request_logs": list(request_logs)
        }
    )

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text from prompt"""
    try:
        generated_text = generator.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        # Log the request and response
        request_logs.appendleft({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": request.prompt,
            "response": generated_text
        })
        
        return {"status": "success", "generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 