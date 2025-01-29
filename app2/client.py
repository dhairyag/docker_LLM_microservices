from flask import Flask, request, render_template
import requests
from urllib.parse import urljoin
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

app = Flask(__name__)

# Configuration
API_CONFIG = {
    "BASE_URL": "http://app1:5000",  # Use container's internal port (5000)
    "GENERATE_ENDPOINT": "/generate"
}

# Configure retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

def get_api_url(endpoint: str) -> str:
    return urljoin(API_CONFIG["BASE_URL"], endpoint)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            payload = {
                'prompt': request.form['prompt'],
                'max_length': int(request.form.get('max_length', 100)),
                'temperature': float(request.form.get('temperature', 0.8)),
                'top_k': int(request.form.get('top_k', 50)),
                'top_p': float(request.form.get('top_p', 0.9)),
                'repetition_penalty': float(request.form.get('repetition_penalty', 1.5))
            }

            response = http.post(
                get_api_url(API_CONFIG["GENERATE_ENDPOINT"]), 
                json=payload,
                timeout=30  # Add timeout
            )
            response.raise_for_status()
            result = response.json()

            return render_template('index.html', 
                                generated_text=result['generated_text'],
                                prompt=payload['prompt'])
        except requests.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            return render_template('index.html', error=error_msg)
        except Exception as e:
            error_msg = f"System Error: {str(e)}"
            return render_template('index.html', error=error_msg)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 