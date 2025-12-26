let API_URL = 'http://localhost:8000';

const elements = {
    processBtn: document.getElementById('processBtn'),
    autoDetectBtn: document.getElementById('autoDetectBtn'),
    chatBtn: document.getElementById('chatBtn'),
    clearBtn: document.getElementById('clearBtn'),
    statusBtn: document.getElementById('statusBtn'),
    videoUrl: document.getElementById('videoUrl'),
    question: document.getElementById('question'),
    response: document.getElementById('response'),
    status: document.getElementById('status'),
    apiUrl: document.getElementById('apiUrl'),
    forceReprocess: document.getElementById('forceReprocess')
};

// Auto-detect current YouTube video URL
async function getCurrentVideoUrl() {
    try {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        if (tab.url && tab.url.includes('youtube.com/watch')) {
            return tab.url;
        }
        return null;
    } catch (error) {
        return null;
    }
}

async function makeRequest(endpoint, data) {
    try {
        updateStatus('Connecting to AI...');
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        updateStatus('Ready');
        return result;
    } catch (error) {
        updateStatus('Connection Error');
        return { success: false, message: `Connection error: ${error.message}` };
    }
}

// Update API_URL when user changes input
elements.apiUrl.addEventListener('change', (e) => {
    API_URL = e.target.value.trim();
});

function updateStatus(message) {
    elements.status.textContent = message;
}

function showResponse(message, isError = false) {
    elements.response.textContent = message;
    elements.response.className = isError ? 'error' : 'success';
}

// Process video with auto-detection
elements.processBtn.addEventListener('click', async () => {
    let video_input = elements.videoUrl.value.trim();
    
    if (!video_input) {
        showResponse('Please enter a YouTube URL or text containing YouTube link.', true);
        return;
    }
    
    showResponse('🔄 Processing video...');
    elements.processBtn.disabled = true;
    updateStatus('Processing...');
    
    const result = await makeRequest('/process', { 
        video_url: video_input,
        force_reprocess: elements.forceReprocess.checked
    });
    
    if (result.success) {
        showResponse(`✅ ${result.message}`);
        updateStatus('Video Ready');
    } else {
        showResponse(`❌ Error: ${result.message || result.detail}`, true);
        updateStatus('Error');
    }
    
    elements.processBtn.disabled = false;
});

// Auto-detect current YouTube video
elements.autoDetectBtn.addEventListener('click', async () => {
    updateStatus('Detecting video...');
    const currentUrl = await getCurrentVideoUrl();
    
    if (currentUrl) {
        elements.videoUrl.value = currentUrl;
        showResponse('✅ YouTube video detected! Click "Process Video" to analyze.');
        updateStatus('Video Detected');
    } else {
        showResponse('❌ No YouTube video found. Please navigate to a YouTube video page.', true);
        updateStatus('No Video Found');
    }
});

// Chat with AI
elements.chatBtn.addEventListener('click', async () => {
    const video_input = elements.videoUrl.value.trim();
    const question = elements.question.value.trim();
    
    if (!video_input || !question) {
        showResponse('Please enter both video URL and question.', true);
        return;
    }
    
    showResponse('🤔 AI is thinking...');
    elements.chatBtn.disabled = true;
    updateStatus('Getting Answer...');
    
    const result = await makeRequest('/chat', { video_url: video_input, question });
    
    if (result.success) {
        showResponse(`💡 ${result.answer}`);
        updateStatus('Answer Ready');
        elements.question.value = ''; // Clear question for next use
    } else {
        showResponse(`❌ Error: ${result.message || result.detail}`, true);
        updateStatus('Error');
    }
    
    elements.chatBtn.disabled = false;
});

// Clear memory
elements.clearBtn.addEventListener('click', async () => {
    if (confirm('Clear all processed video data from memory?')) {
        updateStatus('Clearing...');
        const result = await makeRequest('/clear', {});
        if (result.success) {
            showResponse('🗑️ Memory cleared. You can process new videos.');
            updateStatus('Memory Cleared');
        } else {
            showResponse(`❌ Error clearing memory: ${result.message}`, true);
            updateStatus('Error');
        }
    }
});

// Show status
elements.statusBtn.addEventListener('click', async () => {
    showResponse('📊 Backend is running and ready to process videos.');
    updateStatus('Status Checked');
});

// Auto-detect video on popup open
document.addEventListener('DOMContentLoaded', async () => {
    const currentUrl = await getCurrentVideoUrl();
    if (currentUrl) {
        elements.videoUrl.value = currentUrl;
        updateStatus('Video Auto-Detected');
    } else {
        updateStatus('Ready');
    }
});
