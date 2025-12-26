# Updated backend/rag_pipeline.py with yt-dlp
import subprocess
import json
import os
import tempfile
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prefer langchain_huggingface or langchain_community embeddings to avoid deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings

# Prefer community vectorstores to avoid deprecation warnings
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.vectorstores import Chroma

from langchain.schema import Document
# from langchain_google_genai import ChatGoogleGenerativeAI
import google.genai as genai
from config import Config

class RAGPipeline:
    def __init__(self):
        Config.validate()
        # Delay heavy embedding model initialization until first use.
        # This avoids network/model download at import/startup and
        # allows the FastAPI server to come up even when HuggingFace
        # or sentence-transformers cannot be reached.
        self.embeddings = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300
        )
        # Gemini API setup (new SDK)
        self.genai_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.gemini_model = "gemini-2.5-flash"  # Use Gemini 1.5 Flash model
        # Persistent memory (in-memory cache)
        self.vector_stores = {}
        self.conversation_memory = {}  # Store conversation history per video
        self.vector_store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vector_stores'))
        os.makedirs(self.vector_store_dir, exist_ok=True)

    def ensure_embeddings(self):
        """Initialize embeddings lazily. Returns None on success or an error string on failure."""
        if self.embeddings is not None:
            return None
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return None
        except Exception as e:
            # Keep embeddings as None and return the error message so callers can decide.
            self.embeddings = None
            return str(e)
        
    def extract_video_id(self, text):
        """Extract YouTube video ID from any text containing YouTube URL"""
        import re
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None

    def get_transcript(self, text):
        """Get transcript using YouTube Transcript API (primary) and yt-dlp (fallback)"""
        video_id = self.extract_video_id(text)
        if not video_id:
            return None, "No YouTube URL found in text"
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        print("🎯 Extracting transcript...")
        
        # Try YouTube Transcript API first (more reliable, no rate limiting issues)
        transcript = self._try_youtube_transcript_api(video_id)
        if transcript:
            print("✅ Successfully got transcript using YouTube Transcript API")
            return transcript, None
        
        # Fallback to yt-dlp with conservative approach
        print("⚠️ YouTube Transcript API failed, trying yt-dlp fallback...")
        transcript = self._try_yt_dlp_fallback(video_url)
        if transcript:
            print("✅ Successfully got transcript using yt-dlp fallback")
            return transcript, None

        return None, "No subtitles available for this video. YouTube has strict bot detection. Consider using residential proxies for the YouTube Transcript API."

    def _try_youtube_transcript_api(self, video_id):
        """Try to get transcript using YouTube Transcript API (most reliable)"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

            # Try different language combinations
            language_combinations = [
                ['en'],           # English only
                ['hi'],           # Hindi only
                ['en', 'hi'],     # English first, then Hindi
                ['hi', 'en'],     # Hindi first, then English
            ]

            for languages in language_combinations:
                try:
                    print(f"🎯 Trying YouTube Transcript API with languages: {languages}")
                    
                    # Use YouTubeTranscriptApi without proxy (proxy support removed)
                    ytt_api = YouTubeTranscriptApi()
                    
                    transcript_data = ytt_api.get_transcript(video_id, languages=languages)

                    # Combine all text snippets
                    full_text = ' '.join([snippet['text'] for snippet in transcript_data])

                    print(f"✅ Got transcript with {len(transcript_data)} snippets, {len(full_text)} characters")
                    return full_text.strip()

                except NoTranscriptFound:
                    print(f"❌ No transcript found for languages: {languages}")
                    continue
                except TranscriptsDisabled:
                    print("❌ Transcripts are disabled for this video")
                    return None
                except Exception as e:
                    error_msg = str(e)
                    if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                        print("🚫 YouTube bot detection triggered - proxy might help")
                        return None
                    elif "429" in error_msg or "Too Many Requests" in error_msg:
                        print("🚫 Rate limited by YouTube")
                        return None
                    else:
                        print(f"⚠️ Error with languages {languages}: {e}")
                    continue

            return None

        except ImportError:
            print("❌ YouTube Transcript API not installed")
            return None
        except Exception as e:
            print(f"💥 YouTube Transcript API failed: {e}")
            return None
    def _parse_srt_to_text(self, srt_content):
        """Parse SRT subtitle format to plain text"""
        import re

        # Split into subtitle blocks
        blocks = re.split(r'\n\n+', srt_content.strip())

        text_parts = []
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                # Skip the subtitle number and timestamp lines
                # Take all remaining lines as text
                text_lines = lines[2:]
                # Remove HTML tags and clean up
                clean_text = ' '.join(text_lines)
                clean_text = re.sub(r'<[^>]+>', '', clean_text)
                if clean_text.strip():
                    text_parts.append(clean_text.strip())

        return ' '.join(text_parts)

    def _try_yt_dlp_fallback(self, video_url):
        """Very conservative yt-dlp fallback - try manual then auto subtitles"""
        try:
            print("🔄 yt-dlp fallback: Trying English manual subtitles...")
            
            # Try manual subtitles first
            source_name = "Manual English"
            sub_args = ['--write-sub', '--sub-langs', 'en', '--sub-format', 'vtt']
            
            transcript = self._try_subtitle_source_conservative(video_url, source_name, sub_args)
            if transcript:
                return transcript
            
            print("🔄 yt-dlp fallback: Trying English auto-generated subtitles...")
            
            # Try auto-generated subtitles
            source_name = "Auto English"
            sub_args = ['--write-auto-subs', '--sub-langs', 'en', '--sub-format', 'vtt']
            
            transcript = self._try_subtitle_source_conservative(video_url, source_name, sub_args)
            if transcript:
                return transcript
            
            print("❌ yt-dlp fallback failed - no subtitles available")
            return None
                
        except Exception as e:
            print(f"yt-dlp fallback failed: {e}")
            return None

    def _try_subtitle_source_conservative(self, video_url, source_name, sub_args):
        """Very conservative version - only try once with long timeout"""
        # Allow controlled retries via env var (default 1 for conservative)
        retries = int(os.getenv('YTDLP_RETRIES', '1'))
        for attempt in range(retries):
            try:
                print(f"Trying {source_name} subtitles (attempt {attempt + 1}/{retries})...")

                with tempfile.TemporaryDirectory() as temp_dir:
                    # Simple single-attempt yt-dlp call using a cookies file if available
                    cookie_file = os.getenv('YOUTUBE_COOKIES_PATH')
                    default_cookie_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/cookies/youtube_cookies.txt'))
                    if not cookie_file and os.path.exists(default_cookie_path):
                        cookie_file = default_cookie_path

                    # Build command safely to avoid splitting option/value pairs (Windows paths)
                    base_cmd = [
                        'yt-dlp',
                        '--js-runtime', 'node',
                        '--skip-download',
                        '-o', f'{temp_dir}/%(id)s.%(ext)s',
                    ]

                    if cookie_file and os.path.exists(cookie_file):
                        base_cmd += ['--cookies', cookie_file]

                    download_cmd = base_cmd + [video_url] + sub_args

                    print(f"🔁 Running yt-dlp: {' '.join(download_cmd[:6])} ...")
                    try:
                        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=120)
                    except subprocess.TimeoutExpired:
                        print(f"yt-dlp timeout ({source_name})")
                        result = None

                    result_stdout = result.stdout if result else ''
                    result_stderr = result.stderr if result else ''

                    if result and result.returncode == 0:
                        # Find subtitle file
                        subtitle_file = None
                        video_id = self.extract_video_id(video_url)

                        if '--sub-format' in sub_args and 'vtt' in sub_args:
                            # VTT format
                            lang = sub_args[sub_args.index('--sub-langs') + 1]
                            expected_files = [f"{video_id}.{lang}.vtt", f"{video_id}.vtt"]
                        else:
                            # TXT format
                            lang = sub_args[sub_args.index('--sub-langs') + 1]
                            expected_files = [f"{video_id}.{lang}.txt"]

                        for file in os.listdir(temp_dir):
                            if file in expected_files:
                                subtitle_file = os.path.join(temp_dir, file)
                                break

                        if subtitle_file:
                            with open(subtitle_file, 'r', encoding='utf-8') as f:
                                content = f.read()

                            if subtitle_file.endswith('.vtt'):
                                content = self.parse_vtt_content(content)

                            print(f"✅ yt-dlp got transcript: {len(content)} characters")
                            return content

                    # Check for rate limiting or other errors
                    error_msg = (result_stderr or '').strip()
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        print(f"Rate limited ({source_name}) - giving up")
                        return None
                    else:
                        # Print more verbose stderr to help debugging
                        print(f"yt-dlp failed (attempt {attempt + 1}): {error_msg[:1000]}...")

            except Exception as e:
                print(f"Error with {source_name} on attempt {attempt + 1}: {e}")
                # small backoff
                time.sleep(2)

        # Optional Playwright-based fallback if enabled
        if os.getenv('ENABLE_PLAYWRIGHT_FALLBACK', '0') in ('1', 'true', 'True'):
            try:
                content = self._playwright_subtitle_fallback(video_url)
                if content:
                    return content
            except Exception as e:
                print(f"Playwright fallback error: {e}")

        return None

    def _try_subtitle_source(self, video_url, source_name, sub_args):
        
        max_retries = 2  # Reduced from 3 to be more conservative
        for attempt in range(max_retries):
            try:
                print(f"Trying {source_name} subtitles (attempt {attempt + 1})...")
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Simpler single-attempt run for non-conservative path
                    cookie_file = os.getenv('YOUTUBE_COOKIES_PATH')
                    default_cookie_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/cookies/youtube_cookies.txt'))
                    if not cookie_file and os.path.exists(default_cookie_path):
                        cookie_file = default_cookie_path

                    base_cmd = [
                        'yt-dlp',
                        '--js-runtime', 'node',
                        '--remote-components', 'ejs:github',
                        '--skip-download',
                        '-o', f'{temp_dir}/%(id)s.%(ext)s',
                    ]

                    if cookie_file and os.path.exists(cookie_file):
                        base_cmd += ['--cookies', cookie_file]

                    download_cmd = base_cmd + [video_url] + sub_args

                    print(f"🔁 Running yt-dlp: {' '.join(download_cmd[:6])} ...")
                    try:
                        result = subprocess.run(download_cmd, capture_output=True, text=True)
                    except Exception as e:
                        print(f"yt-dlp run error: {e}")
                        result = None

                    if result and result.returncode == 0:
                        # Try to find the subtitle file
                        subtitle_file = None
                        expected_extensions = []
                        
                        if '--sub-format' in sub_args and 'vtt' in sub_args:
                            # VTT format (manual subtitles)
                            lang = sub_args[sub_args.index('--sub-langs') + 1]
                            expected_extensions = [f'.{lang}.vtt', '.vtt']
                        elif '--sub-format' in sub_args and 'txt' in sub_args:
                            # TXT format (auto-generated)
                            lang = sub_args[sub_args.index('--sub-langs') + 1]
                            expected_extensions = [f'.{lang}.txt']
                        
                        for file in os.listdir(temp_dir):
                            for ext in expected_extensions:
                                if file.endswith(ext):
                                    subtitle_file = os.path.join(temp_dir, file)
                                    break
                            if subtitle_file:
                                break
                        
                        if subtitle_file:
                            with open(subtitle_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if subtitle_file.endswith('.vtt'):
                                content = self.parse_vtt_content(content)
                            
                            print(f"✅ Successfully got transcript from {source_name}")
                            return content
                    
                    # Check for rate limiting
                    error_msg = result.stderr.strip()
                    if "429" in error_msg and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # Increased from 5 to 10 seconds
                        print(f"Rate limited ({source_name}), waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    elif attempt < max_retries - 1:
                        print(f"{source_name} failed, trying next source...")
                        return None
                    
            except Exception as e:
                print(f"Error with {source_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Increased from 2 to 3 seconds
                    continue
        
        return None

    def parse_vtt_content(self, vtt_content):
        """Parse VTT subtitle format to extract plain text"""
        lines = vtt_content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and
                not line.startswith('NOTE') and
                '-->' not in line and
                not line.isdigit() and
                not line.startswith('<')):
                # Remove VTT tags
                import re
                line = re.sub(r'<[^>]+>', '', line)
                if line:
                    text_lines.append(line)
        
        return ' '.join(text_lines)

    def _playwright_subtitle_fallback(self, video_url):
        """Optional Playwright fallback: render page, extract player response and fetch caption URL."""
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            print("Playwright not installed - skip playwright fallback. Install with: pip install playwright && playwright install")
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.goto(video_url, wait_until='networkidle', timeout=30000)

                # Try retrieving the initial player response object
                player_response = page.evaluate("""() => {
                    try {
                        if (window.ytInitialPlayerResponse) return JSON.stringify(window.ytInitialPlayerResponse);
                        if (window.ytplayer && window.ytplayer.config && window.ytplayer.config.args && window.ytplayer.config.args.player_response) return window.ytplayer.config.args.player_response;
                        return null;
                    } catch(e) { return null; }
                }""")

                browser.close()

                if not player_response:
                    print("Playwright: no player response found")
                    return None

                import json as _json, re
                try:
                    player = _json.loads(player_response)
                except Exception:
                    # attempt to extract JSON object from string
                    m = re.search(r'\{.+\}', player_response, re.S)
                    if not m:
                        print("Playwright: could not parse player response")
                        return None
                    player = _json.loads(m.group(0))

                captions = player.get('captions') or {}
                tlist = captions.get('playerCaptionsTracklistRenderer') or {}
                tracks = tlist.get('captionTracks') or []
                if not tracks:
                    print("Playwright: no caption tracks available")
                    return None

                # Prefer English track if available
                track = next((t for t in tracks if t.get('languageCode','').startswith('en')), tracks[0])
                base_url = track.get('baseUrl')
                if not base_url:
                    print("Playwright: caption track has no baseUrl")
                    return None

                try:
                    import requests
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    r = requests.get(base_url, timeout=20, headers=headers)
                    if r.status_code == 200:
                        text = r.text
                        # Many caption tracks are in vtt/xml formats
                        if base_url.endswith('.vtt') or 'vtt' in r.headers.get('Content-Type',''):
                            return self.parse_vtt_content(text)
                        else:
                            return text
                    else:
                        print(f"Playwright: failed to fetch captions, status {r.status_code}")
                        return None
                except Exception as e:
                    print(f"Playwright: error fetching caption URL: {e}")
                    return None

        except Exception as e:
            print(f"Playwright fallback error: {e}")
            return None

    def process_video(self, video_url, force_reprocess=False):
        """Process video and store in persistent memory and disk"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {"success": False, "message": "No valid YouTube URL found"}
        # Check if already processed (memory) - skip if force_reprocess is False
        if not force_reprocess and video_id in self.vector_stores:
            return {"success": True, "message": "Video already processed - using cached data"}
        # Remove from memory if force reprocessing
        if force_reprocess and video_id in self.vector_stores:
            del self.vector_stores[video_id]
            # Also clear conversation memory for this video
            if video_id in self.conversation_memory:
                del self.conversation_memory[video_id]
        # Check if already processed (disk)
        # ChromaDB persistent check handled below
        # Get transcript
        transcript, error = self.get_transcript(video_url)
        if error and not transcript:
            return {"success": False, "message": error}
        
        # Create chunks and persistent vector store (persist per-video)
        # Ensure embeddings are available before creating vector stores
        err = self.ensure_embeddings()
        if err:
            return {"success": False, "message": f"Failed to initialize embeddings: {err}"}

        texts = self.splitter.split_text(transcript)
        docs = [Document(page_content=t) for t in texts]
        chroma_path = os.path.join(self.vector_store_dir, video_id)
        os.makedirs(chroma_path, exist_ok=True)

        try:
            vector_store = Chroma.from_documents(
                docs,
                embedding=self.embeddings,
                persist_directory=chroma_path,
            )
            # Chroma automatically persists documents for modern versions; skip manual persist.

            self.vector_stores[video_id] = vector_store
        except Exception as e:
            print(f"⚠️ Failed to create persistent Chroma store: {e}")
            # Fallback to in-memory store
            vector_store = Chroma.from_documents(docs, embedding=self.embeddings)
            self.vector_stores[video_id] = vector_store
        # ChromaDB persist handled above
        success_message = f"Successfully processed and cached {len(docs)} chunks"
        if error:
            success_message += f" - {error}"
        return {"success": True, "message": success_message}

    def get_answer(self, video_url, question):
        """Get answer using persistent memory or disk for the video"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {"success": False, "message": "No valid YouTube URL found"}
        # Check persistent memory
        if video_id not in self.vector_stores:
            # Try loading from ChromaDB persistent
            chroma_path = os.path.join(self.vector_store_dir, video_id)
            if os.path.exists(chroma_path):
                try:
                    # Ensure embeddings are available before loading vector store
                    err = self.ensure_embeddings()
                    if err:
                        return {"success": False, "message": f"Failed to initialize embeddings: {err}"}

                    vector_store = Chroma(persist_directory=chroma_path, embedding_function=self.embeddings)
                    self.vector_stores[video_id] = vector_store
                except Exception as e:
                    return {"success": False, "message": f"Failed to load vector store from disk: {str(e)}"}
            else:
                return {"success": False, "message": "Video not processed yet. Please process the video first."}
        # Use cached vector store
        vector_store = self.vector_stores[video_id]
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        def format_docs(docs_list):
            return "\n\n".join(getattr(d, 'page_content', str(d)) for d in docs_list)

        # Retrieve context using stable retriever API
        try:
            docs_list = retriever.get_relevant_documents(question)
        except Exception:
            # Fallback for older/newer versions
            if hasattr(retriever, 'get_relevant_documents'):
                docs_list = retriever.get_relevant_documents(question)
            elif hasattr(retriever, 'get_documents'):
                docs_list = retriever.get_documents(question)
            else:
                docs_list = []

        context = format_docs(docs_list)
        
        # Get conversation history for this video
        if video_id not in self.conversation_memory:
            self.conversation_memory[video_id] = []
        
        conversation_history = self.conversation_memory[video_id]
        history_text = ""
        if conversation_history:
            history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])  # Last 3 exchanges
        
        prompt = f"""You are a helpful AI assistant that answers questions about YouTube videos.

CRITICAL LANGUAGE RULE: ALWAYS respond in ENGLISH unless the user's question is explicitly in Hindi. 
- If question is in English → Answer in English
- If question is in Hindi → Answer in Hindi  
- NEVER switch languages mid-conversation
- IGNORE transcript language for response language - base it ONLY on the user's question language

- Answer ONLY using information from the video transcript provided below
- Keep answers SHORT and SIMPLE - avoid unnecessary details
- PROVIDE HELPFUL, DETAILED ANSWERS when asked for explanations - use all available context from the transcript
- If you don't have enough information from this video to answer that question, say so clearly
- Be accurate and specific in your responses
- When explaining concepts, break them down step-by-step if the transcript supports it
- CONSIDER the conversation history when answering to provide better context{history_text}

VIDEO TRANSCRIPT:
{context}

QUESTION: {question}

ANSWER:"""
        try:
            response = self.genai_client.models.generate_content(model=self.gemini_model, contents=prompt)
            if response and hasattr(response, 'text'):
                answer = response.text
                # Store this Q&A in conversation memory
                self.conversation_memory[video_id].append((question, answer))
                # Keep only last 10 exchanges to avoid memory bloat
                if len(self.conversation_memory[video_id]) > 10:
                    self.conversation_memory[video_id] = self.conversation_memory[video_id][-10:]
                return {"success": True, "answer": answer}
            else:
                return {"success": False, "message": "No response from Gemini API"}
        except Exception as e:
            return {"success": False, "message": f"Error generating answer: {str(e)}"}
    
    def get_video_info(self, video_url):
        """Get information about a processed video including language"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {"success": False, "message": "No valid YouTube URL found"}
        
        if video_id not in self.vector_stores:
            # Try loading from disk
            chroma_path = os.path.join(self.vector_store_dir, video_id)
            if os.path.exists(chroma_path):
                try:
                    vector_store = Chroma(persist_directory=chroma_path, embedding_function=self.embeddings)
                    self.vector_stores[video_id] = vector_store
                except Exception as e:
                    return {"success": False, "message": f"Failed to load vector store: {str(e)}"}
            else:
                return {"success": False, "message": "Video not processed yet"}
        
        # Get language from stored data
        try:
            docs = self.vector_stores[video_id].similarity_search("", k=1)
            if docs:
                sample_text = docs[0].page_content[:1000]
                detected_lang = self.detect_transcript_language(sample_text)
                return {"success": True, "video_id": video_id, "language": detected_lang, "status": "processed"}
        except Exception as e:
            return {"success": False, "message": f"Error detecting language: {str(e)}"}
        
    def get_processed_videos(self):
        """Get list of processed video IDs"""
        return list(self.vector_stores.keys())
    
    def clear_memory(self):
        """Clear all processed videos from memory"""
        self.vector_stores.clear()
        self.conversation_memory.clear()
        return {"success": True, "message": "All video data and conversation history cleared from memory"}
