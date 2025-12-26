# YouTube AI Chatbot

Chat with YouTube videos using Gemini AI.

## Setup

1. Get Gemini API key from Google AI Studio
2. Add to .env file: `GOOGLE_API_KEY=your_key_here`
3. Install: `pip install -r requirements.txt`
4. Run: `cd backend && python app.py`
5. Load extension in Chrome from extension/ folder
6. (Recommended) Set an API key to protect the backend before deployment:

	 - Add `API_KEY` to your environment or `.env` file.
	 - The server will require the header `x-api-key: <API_KEY>` on requests (except `/health` and docs).

	 Example curl using the API key:

	 ```bash
	 curl -H "x-api-key: your_api_key_here" -X POST http://localhost:8000/process \
		 -H "Content-Type: application/json" \
		 -d '{"video_url":"https://youtu.be/dQw4w9WgXcQ","force_reprocess":false}'
	 ```

## Production Deployment (Recommended)

YouTube has strict bot detection that may cause subtitle extraction to fail. For production use:

### Note on proxy usage

This project previously documented a residential-proxy workflow to bypass YouTube bot detection. Proxy-based subtitle retrieval proved fragile for many deployments and is not required for most users. The current codebase no longer uses the `YOUTUBE_PROXY_USERNAME` / `YOUTUBE_PROXY_PASSWORD` environment variables; subtitle extraction will use the YouTube Transcript API first and fall back to `yt-dlp` or Playwright (if enabled).

If you still need proxies for production-scale deployments, consider using a managed proxy service and adapt the transcript-fetching logic yourself. Be aware this increases operational complexity and cost.

### Option 2: YouTube Data API (Requires OAuth - Complex)

**Note:** The YouTube Data API v3 captions download requires OAuth 2.0, not API keys. This is more complex to set up and not recommended for simple applications.

### Option 3: Accept Limitations

For development/testing, the current setup works but may occasionally fail due to YouTube's bot detection.

## Usage

1. Go to YouTube video
2. Click extension icon
3. Process video
4. Ask questions about the content

**Persistence & Docker**

- Persist Chroma vector stores to a mounted volume so processed videos survive restarts. The app stores per-video data under `data/vector_stores/<video_id>` by default.
- Example Docker run with a host volume for vector stores and envs:

```bash
docker build -t youtube-chatbot .
docker run -d \
	-e GOOGLE_API_KEY=your_key_here \
	-e YOUTUBE_API_KEY=your_key_here \
	-p 8000:8000 \
	-v "$(pwd)/data/vector_stores:/app/data/vector_stores" \
	--name youtube-chatbot youtube-chatbot
```

Make sure to store secrets securely in your production environment (not in plain `.env` files inside the container).
