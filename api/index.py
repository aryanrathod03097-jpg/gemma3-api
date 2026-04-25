from flask import Flask, request, jsonify
import requests
import time
from datetime import datetime

app = Flask(__name__)

# Configuration
BACKEND_API_URL = "http://38.12.5.106:11434/api/chat"
TIMEOUT = 60  # seconds for Gemma model
API_VERSION = "1.0.0"
API_NAME = "Gemma 3"
OWNER = "AlexApiForest"
MODEL_NAME = "gemma3:1b"

# Social Media Links
SOCIAL_LINKS = {
    "youtube": "@Kaiiddo",
    "github": "ProKaiiddo",
    "telegram": "@xKaiiddo",
    "twitter": "@Kaiiddo"
}

# Custom Error Classes
class Gemma3Error(Exception):
    def __init__(self, message, status_code, error_code):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)

class ValidationError(Gemma3Error):
    def __init__(self, message):
        super().__init__(message, 400, "VALIDATION_ERROR")

class BackendAPIError(Gemma3Error):
    def __init__(self, message):
        super().__init__(message, 502, "BACKEND_API_ERROR")

class TimeoutError(Gemma3Error):
    def __init__(self):
        super().__init__("Request timeout - Gemma 3 model took too long to respond", 504, "TIMEOUT_ERROR")

class RateLimitError(Gemma3Error):
    def __init__(self):
        super().__init__("Too many requests - Please try again later", 429, "RATE_LIMIT_ERROR")

class ModelError(Gemma3Error):
    def __init__(self, message):
        super().__init__(message, 500, "MODEL_ERROR")

# Error Handler
@app.errorhandler(Gemma3Error)
def handle_gemma3_error(error):
    response = {
        "status": "error",
        "error": {
            "code": error.error_code,
            "message": error.message,
            "type": error.__class__.__name__
        },
        "model": MODEL_NAME,
        "timestamp": error.timestamp,
        "api_version": API_VERSION
    }
    return jsonify(response), error.status_code

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": {
            "code": "NOT_FOUND",
            "message": "The requested endpoint does not exist",
            "type": "NotFoundError"
        },
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": API_VERSION
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "type": "InternalServerError"
        },
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": API_VERSION
    }), 500

# Helper function to format response
def format_success_response(backend_data, user_message):
    return {
        "status": "success",
        "model": backend_data.get("model", MODEL_NAME),
        "created_at": backend_data.get("created_at", datetime.utcnow().isoformat()),
        "message": {
            "role": backend_data.get("message", {}).get("role", "assistant"),
            "content": backend_data.get("message", {}).get("content", "")
        },
        "question": user_message,
        "done": backend_data.get("done", True),
        "done_reason": backend_data.get("done_reason", "stop"),
        "performance": {
            "total_duration_ms": round(backend_data.get("total_duration", 0) / 1000000, 2),
            "load_duration_ms": round(backend_data.get("load_duration", 0) / 1000000, 2),
            "prompt_eval_count": backend_data.get("prompt_eval_count", 0),
            "prompt_eval_duration_ms": round(backend_data.get("prompt_eval_duration", 0) / 1000000, 2),
            "eval_count": backend_data.get("eval_count", 0),
            "eval_duration_ms": round(backend_data.get("eval_duration", 0) / 1000000, 2)
        },
        "credits": {
            "api_name": API_NAME,
            "owner": OWNER,
            "model": MODEL_NAME
        },
        "api_version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

# Routes
@app.route('/', methods=['GET'])
def home():
    """API Info Endpoint"""
    return jsonify({
        "status": 200,
        "api_name": API_NAME,
        "version": API_VERSION,
        "model": MODEL_NAME,
        "owner": OWNER,
        "social_links": SOCIAL_LINKS,
        "endpoints": {
            "chat": "/chat?q=your_message (GET method)",
            "chat_post": "/chat (POST method with JSON body)",
            "info": "/"
        },
        "usage_examples": {
            "get_method": "/chat?q=Hello, how are you?",
            "post_method": {
                "endpoint": "/chat",
                "body": {
                    "q": "Hello, how are you?"
                }
            }
        },
        "credits": f"Created by {OWNER} | Powered by Google {MODEL_NAME}",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat Endpoint - Supports GET and POST methods"""
    
    user_message = ""
    
    # GET request - Get question from query parameter
    if request.method == 'GET':
        user_message = request.args.get('q', '').strip()
    
    # POST request - Get question from JSON body or form data
    elif request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            user_message = data.get('q', data.get('message', data.get('content', ''))).strip()
        else:
            user_message = request.form.get('q', request.form.get('message', '')).strip()
    
    # Validate user message
    if not user_message:
        raise ValidationError("Message is required. Use 'q' parameter for GET or JSON body with 'q' key for POST")
    
    if len(user_message) > 2000:
        raise ValidationError("Message too long - Maximum 2000 characters allowed")
    
    # Prepare request to backend
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False
    }
    
    # Call backend API
    try:
        start_time = time.time()
        
        response = requests.post(
            BACKEND_API_URL,
            json=payload,
            timeout=TIMEOUT,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': f'{API_NAME}/1.0'
            }
        )
        
        # Check for timeout
        if time.time() - start_time > TIMEOUT:
            raise TimeoutError()
        
        # Check HTTP status
        if response.status_code != 200:
            raise BackendAPIError(f"Backend returned status code {response.status_code}")
        
        # Parse response
        try:
            data = response.json()
        except ValueError:
            raise BackendAPIError("Invalid JSON response from Gemma 3 backend")
        
        # Validate response structure
        if "message" not in data or "content" not in data.get("message", {}):
            raise ModelError("Invalid response structure from model")
        
        # Return formatted success response
        return jsonify(format_success_response(data, user_message)), 200
        
    except requests.exceptions.Timeout:
        raise TimeoutError()
        
    except requests.exceptions.ConnectionError:
        raise BackendAPIError("Unable to connect to Gemma 3 backend server")
        
    except requests.exceptions.RequestException as e:
        raise BackendAPIError(f"Backend API error: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Health Check Endpoint"""
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "api_version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel
app = app
