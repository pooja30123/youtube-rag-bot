# YouTube RAG Bot

A FastAPI-based chatbot that lets you chat with YouTube videos using Retrieval-Augmented Generation (RAG) and Gemini AI.  
Ask questions about any YouTube video and get answers based on its content!

---

## 🚀 Features

- **Chat with YouTube Videos:** Ask questions and get answers based on the video’s transcript and content.
- **Gemini AI Integration:** Uses Gemini AI for natural language understanding and response generation.
- **Cookie Management:** Supports uploading cookies for accessing restricted or private videos.
- **API Key Security:** Optionally protect your backend with an API key.
- **Browser Extension:** Chrome extension for easy video processing and chatting.

---

## 📝 How It Works

1. **Process a Video:**  
   The backend fetches the transcript and processes the video using RAG techniques.
2. **Ask Questions:**  
   You can ask questions about the processed video, and the bot will answer using the video’s content.
3. **Cookie Support:**  
   Upload your YouTube cookies to access private or age-restricted videos.
4. **API Security:**  
   Protect your API endpoints with an API key for secure access.

---

## ⚙️ Getting Started

1. **Get a Gemini API Key:**  
   - Sign up at Google AI Studio and obtain your Gemini API key.

2. **Configure Environment:**  
   - Create a `.env` file in the project root:
     ```
     GOOGLE_API_KEY=your_key_here
     ```
   - (Optional) Add an API key for backend protection:
     ```
     API_KEY=your_custom_api_key
     ```

3. **Install Dependencies:**  
   ```
   pip install -r requirements.txt
   ```

4. **Run the Backend:**  
   ```
   cd backend
   python app.py
   ```

5. **Use the Chrome Extension:**  
   - Load the `extension/` folder as an unpacked extension in Chrome.
   - Go to any YouTube video, click the extension icon, and start chatting!

---

## 🛡️ API Key Usage (Recommended)

To protect your backend, set an `API_KEY` in your environment or `.env` file.  
All requests (except `/health` and docs) will require the header:

```
x-api-key: <API_KEY>
```

**Example cURL:**
```bash
curl -H "x-api-key: your_api_key_here" -X POST http://localhost:8000/process \
     -H "Content-Type: application/json" \
     -d '{"video_url":"https://youtu.be/dQw4w9WgXcQ","force_reprocess":false}'
```

---

## 🧩 Usage Flow

1. **Go to a YouTube video.**
2. **Click the extension icon.**
3. **Process the video.**
4. **Ask questions about the video content.**

---

## 📁 Data Persistence

- Processed video data is stored under `data/vector_stores/<video_id>`.
- Uploaded cookies are stored in `data/cookies/`.

---

## ℹ️ Notes

- For best results, use valid cookies for private or age-restricted videos.
- Store your API keys and secrets securely.
- This project is for educational and research purposes.

---

## 👩‍💻 Author

[Your Name]

---

Enjoy chatting with YouTube videos!