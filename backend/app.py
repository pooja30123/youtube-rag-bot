from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import uvicorn
import json
import asyncio

app = FastAPI(title="YouTube Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Simple API key middleware. If `API_KEY` env var is set, requests must include
    header `x-api-key: <API_KEY>`. If `API_KEY` is not set, middleware is a no-op.
    """
    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.getenv("API_KEY")

    async def dispatch(self, request, call_next):
        # If no API key configured, don't enforce auth
        if not self.api_key:
            return await call_next(request)

        # Allow unauthenticated access to health and docs
        allowed_paths = {"/health", "/openapi.json", "/docs", "/redoc"}
        if request.url.path in allowed_paths:
            return await call_next(request)

        incoming = request.headers.get("x-api-key")
        if not incoming or incoming != self.api_key:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        return await call_next(request)


app.add_middleware(APIKeyAuthMiddleware)

rag = RAGPipeline()

class VideoRequest(BaseModel):
    video_url: str
    force_reprocess: bool = False

class ChatRequest(BaseModel):
    video_url: str
    question: str

@app.post("/process")
async def process_video(request: Request):
    try:
        body = await request.json()
        print("🔍 /process received:", json.dumps(body, indent=2))

        # Validate request
        req = VideoRequest(**body)
        print("✅ Request validated")

        # Run processing in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(rag.process_video, req.video_url, req.force_reprocess)
        print("📊 RAG result:", result)

        if not result["success"]:
            print("❌ RAG failed:", result.get("message"))
            raise HTTPException(status_code=400, detail=result.get("message"))

        print("✅ Success:", result.get("message"))
        return result

    except Exception as e:
        print("💥 ERROR:", str(e))
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.post('/upload_cookies')
async def upload_cookies(file: UploadFile = File(...)):
    """Upload a cookies.txt file (Netscape format) and set YOUTUBE_COOKIES_PATH to its location."""
    try:
        import os
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        cookies_dir = os.path.join(data_dir, 'cookies')
        os.makedirs(cookies_dir, exist_ok=True)

        dest_path = os.path.join(cookies_dir, file.filename)
        with open(dest_path, 'wb') as f:
            contents = await file.read()
            f.write(contents)

        # Set environment variable for current process
        os.environ['YOUTUBE_COOKIES_PATH'] = dest_path

        print(f"✅ Uploaded cookies to {dest_path}")
        return {"success": True, "path": dest_path, "message": "Cookies uploaded and applied"}

    except Exception as e:
        print("💥 ERROR uploading cookies:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/cookie_mode')
async def cookie_mode(request: Request):
    """Set cookie usage mode. JSON body: {"use_browser": true|false}
    When `use_browser` is true, the server will attempt to read cookies from the browser (may fail on Windows).
    """
    try:
        body = await request.json()
        use_browser = bool(body.get('use_browser', False))
        import os
        os.environ['YTDLP_USE_BROWSER_COOKIES'] = '1' if use_browser else '0'
        return {"success": True, "use_browser": use_browser}
    except Exception as e:
        print("💥 ERROR setting cookie mode:", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/test_cookies')
async def test_cookies(request: Request):
    """Run yt-dlp --list-subs for the provided video URL using the current cookie config and return logs."""
    try:
        body = await request.json()
        video_url = body.get('video_url')
        if not video_url:
            raise HTTPException(status_code=400, detail='video_url required')

        import os, subprocess
        cookie_file = os.getenv('YOUTUBE_COOKIES_PATH')

        base_cmd = ['yt-dlp', '--skip-download', '--list-subs', video_url, '--js-runtimes', 'node']
        if cookie_file:
            cmd = base_cmd[:-1] + ['--cookies', cookie_file, base_cmd[-1]]
        else:
            cmd = base_cmd

        print(f"🔁 Running test yt-dlp command: {' '.join(cmd[:5])} ...")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {
                'success': proc.returncode == 0,
                'returncode': proc.returncode,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'used_cookie_file': cookie_file or None
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'message': 'yt-dlp timed out', 'used_cookie_file': cookie_file or None}

    except HTTPException:
        raise
    except Exception as e:
        print('💥 ERROR running test_cookies:', str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("🔍 /chat received:", json.dumps(body, indent=2))
        
        # Validate request
        req = ChatRequest(**body)
        print("✅ Request validated")
        
        # Get answer (run in thread to avoid blocking event loop)
        result = await asyncio.to_thread(rag.get_answer, req.video_url, req.question)
        print("📊 RAG result:", result)
        
        if not result or not result.get("success"):
            message = result.get("message", "Unknown error") if result else "No result returned"
            print("❌ RAG failed:", message)
            raise HTTPException(status_code=400, detail=message)
            
        print("✅ Success")
        return result
        
    except Exception as e:
        print("💥 ERROR:", str(e))
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/clear")
async def clear_memory():
    try:
        print("🔍 /clear received")
        
        # Clear memory
        result = rag.clear_memory()
        print("📊 Clear result:", result)
        
        print("✅ Memory cleared")
        return result
        
    except Exception as e:
        print("💥 ERROR:", str(e))
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
