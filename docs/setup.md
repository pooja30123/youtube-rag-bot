#!/bin/bash

# Main project folders
mkdir -p youtube-chatbot/backend
mkdir -p youtube-chatbot/extension
mkdir -p youtube-chatbot/data
mkdir -p youtube-chatbot/docs

# Backend structure
touch youtube-chatbot/backend/app.py
touch youtube-chatbot/backend/config.py
touch youtube-chatbot/backend/rag_pipeline.py

# Extension structure
touch youtube-chatbot/extension/manifest.json
touch youtube-chatbot/extension/popup.html
touch youtube-chatbot/extension/popup.js
touch youtube-chatbot/extension/content.js
touch youtube-chatbot/extension/styles.css

# Docs
touch youtube-chatbot/docs/setup.md

# Root project files
touch youtube-chatbot/requirements.txt
touch youtube-chatbot/.env
touch youtube-chatbot/.gitignore
touch youtube-chatbot/README.md

echo "✅ File structure created"
