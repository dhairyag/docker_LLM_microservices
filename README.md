# SmolLM2 Docker Microservices

A containerized deployment of SmolLM2 language model using a microservices architecture. The project demonstrates Docker container communication with one container serving the model and another handling user interactions.

## Architecture

### App1: Model Server (FastAPI)
- **Purpose**: Serves the SmolLM2 language model through a RESTful API
- **Tech Stack**:
  - FastAPI for API development
  - PyTorch for model inference
  - Transformers library for tokenization
  - Jinja2 for monitoring interface
- **Features**:
  - RESTful endpoint for text generation
  - Real-time request monitoring dashboard
  - Configurable generation parameters
  - Health checks for container orchestration
- **Endpoints**:
  - `/generate` (POST): Generate text from prompt
  - `/` (GET): Monitoring interface
  - `/docs`: API documentation (Swagger UI)

### App2: Web Client (Flask)
- **Purpose**: Provides user interface and handles communication with model server
- **Tech Stack**:
  - Flask for web server
  - Requests library with retry mechanism
  - Bootstrap for UI styling
- **Features**:
  - Web form for prompt input
  - Configurable generation parameters
  - Error handling with user feedback
  - Automatic retry on connection failures

## Docker Configuration

### Container Communication
- Internal network: `app-network`
- Port mappings:
  - App1: `5001:5000` (Host:Container)
  - App2: `8000:8000`
- Health checks implemented for container orchestration

### Volume Mounts
- Model files mounted at `/app/smollm2_model_final`
- Templates for both services

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-repo/smollm2-docker-microservices.git
cd smollm2-docker-microservices
```

2. Place your trained model files in `smollm2_model_final/`:
The model for this project can be downloaded from [Hugging Face](https://huggingface.co/spaces/dhairyashil/SmolLM2_GroundUp/tree/main/smollm2_model_final).
```bash
smollm2_model_final/
├── config.json
├── pytorch_model.bin
└── tokenizer/
```

3. Build and run containers:
```bash
docker-compose up --build
```

4. Access the services:
- Web Interface: http://localhost:8000
- API Monitor: http://localhost:5001
- API Documentation: http://localhost:5001/docs

## API Parameters

- `prompt`: Input text to generate from
- `max_length`: Maximum length of generated text (default: 100)
- `temperature`: Sampling temperature (default: 0.8)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `repetition_penalty`: Penalty for repeated tokens (default: 1.5)

## Project Structure
```bash
.
├── app1/ # Model Server
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── server.py # FastAPI server
│ ├── generator.py # Text generation logic
│ ├── model.py # Model architecture
│ └── templates/ # Monitoring interface
├── app2/ # Web Client
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── client.py # Flask server
│ └── templates/ # Web interface
├── docker-compose.yml # Container orchestration
└── smollm2_model_final/ # (Downloaded from Hugging Face as explained above)
```

## Technical Implementation Details

### Container Communication
- Uses Docker's internal DNS for service discovery
- Implements retry mechanism for reliability
- Health checks ensure service availability

### Error Handling
- Graceful degradation on service failures
- Detailed error messages for debugging
- Automatic retry for transient failures

### Monitoring
- Real-time request logging
- Response preview in monitoring interface
- Auto-refresh functionality
