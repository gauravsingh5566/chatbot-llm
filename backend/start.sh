#!/bin/bash
source /home/ubuntu/chatbot-llm/backend/llama-chatbot-env/bin/activate  # Activate your virtual environment
export FLASK_APP=app.py  # Replace with your actual Flask app name if different
export FLASK_ENV=production
gunicorn -w 1 -b 0.0.0.0:5000 app:app 
