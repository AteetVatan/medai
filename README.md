# 🤖 medAI MVP 🤖 - Clinical Speech & Documentation Platform

A real-time clinical intake and documentation backend with speech-to-text, medical entity recognition, LLM summarization, and translation capabilities.

## 🏗️ Architecture

### Core Components

- **STT Service**: Whisper (openai/whisper-large-v3) and Together (openai/whisper-large-v3)
- **NER Service**: External microservice architecture for medical entity recognition
- **LLM Service**: Mistral 7B with cost-effective fallbacks
- **Translation Service**: NLLB-200 primary, DeepL fallback
- **Storage Service**: Supabase Postgres + S3
- **Clinical Agent**: Agno-based orchestration pipeline

### Performance Guardrails

- STT partials ≤ 300ms, finalization < 2s
- LLM summarization p95 < 1.8s (Mistral) or fallback
- Translation p95 < 1.0s (NLLB) or DeepL fallback
- Temperature = 0.0 everywhere for deterministic output

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- API keys for services (see Configuration)

### Local Development

1. **Clone and setup**:
```bash
git clone <repository>
cd medAI/project/server
cp env.example .env
# Edit .env with your API keys
```

2. **Start with Docker Compose**:
```bash
docker-compose up -d
```

3. **Or run locally**:
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

4. **Access the application**:
- API: http://localhost:8000
- Frontend: http://localhost:8000 (served by FastAPI)
- API Docs: http://localhost:8000/docs

### Configuration

Copy `env.example` to `.env` and configure:

```bash
# Required API Keys
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Optional LLM Providers
MISTRAL_API_KEY=your_key
OPENROUTER_API_KEY=your_key
TOGETHER_API_KEY=your_key
```

## 📁 Project Structure

```
src/
├── agents/
│   └── clinical_intake_agent.py      # Agno agent orchestration
├── services/
│   ├── stt_service.py                # Speech-to-text with fallbacks
│   ├── ner_service.py                # Medical entity extraction
│   ├── llm_service.py                # LLM summarization
│   ├── translation_service.py        # Translation (NLLB/DeepL)
│   └── storage_service.py            # Supabase integration
├── api/
│   ├── main.py                       # FastAPI REST endpoints
│   └── ws.py                         # WebSocket handlers
├── utils/
│   ├── config.py                     # Pydantic settings
│   └── logging.py                    # Structured logging
└── db/
    ├── schema.sql                    # Database schema
    └── rls.sql                       # Row-level security

frontend/
├── index.html                        # Minimal UI
└── script.js                         # WebSocket client

tests/
├── test_stt.py                       # STT service tests
├── test_ner.py                       # NER service tests
├── test_llm.py                       # LLM service tests
├── test_translation.py               # Translation tests
└── test_clinical_agent.py            # Agent orchestration tests
```

## 🔧 API Endpoints

### REST API

- `GET /health` - Health check
- `POST /patients` - Create patient
- `GET /patients?q=search` - Search patients
- `POST /encounters` - Create encounter
- `GET /encounters/{id}` - Get encounter
- `POST /clinical-intake` - Process audio intake
- `GET /audio/{id}/download` - Get audio download URL

### WebSocket

- `WS /ws/{session_id}` - Real-time audio streaming
- Messages: `start_session`, `audio_chunk`, `end_session`

## 🎤 Frontend Usage

1. **Start Session**: Enter encounter ID and select task type
2. **Record Audio**: Click record button, speak into microphone
3. **Real-time Updates**: See partial transcriptions as you speak
4. **View Results**: Structured clinical notes with entities and summary
5. **Save/Download**: Export results as JSON

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific service tests
pytest tests/test_stt.py
pytest tests/test_ner.py
pytest tests/test_llm.py
pytest tests/test_translation.py
pytest tests/test_clinical_agent.py

# With coverage
pytest --cov=src tests/
```

## 🐳 Deployment

### Local Development
```bash
docker-compose up -d
```

### Production (CPU)
```bash
docker build -t medai-mvp --target production .
docker run -p 8000:8000 --env-file .env medai-mvp
```

### GPU Deployment (RunPod)
```bash
docker build -t medai-mvp --target gpu .
# Deploy to RunPod with GPU support
```

### Environment Variables

Key production settings:

```bash
# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
ENABLE_CACHING=true

# Security
ENABLE_PII_STRIPPING=true
AUDIT_LOG_RETENTION_DAYS=90

# Logging
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true
```

## 📊 Monitoring

### Health Checks

- `GET /health` - Overall system health
- Individual service health via agent health check

### Metrics

- Latency tracking for all operations
- Fallback usage monitoring
- Error rate monitoring
- Performance guardrail alerts

### Logging

- Structured JSON logging
- Request ID tracking
- Compliance audit trails
- Performance metrics

## 🔒 Security & Compliance

### Data Protection

- PII stripping before LLM calls
- Signed URLs for S3 access
- Row-level security (RLS) on Supabase
- Full audit logging

### Access Control

- Organization-based data isolation
- User role-based permissions
- Session-based WebSocket connections
- API key rotation support

## 🚀 Performance Optimization

### Latency Strategies

1. **Chunked Audio Streaming**: 1-2s chunks for real-time processing
2. **Async/Await**: Non-blocking I/O throughout
3. **Warm-up**: Service initialization on startup
4. **Parallel Processing**: Concurrent service calls where possible
5. **Caching**: Frequent system prompts and responses
6. **Fallback Triggers**: Automatic provider switching on latency thresholds

### Scaling

- Horizontal scaling with multiple workers
- Database connection pooling
- Redis caching layer
- CDN for static assets

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Services

1. Create service class in `src/services/`
2. Add corresponding tool in `src/agents/`
3. Update agent orchestration
4. Add tests in `tests/`
5. Update health checks

### Database Migrations

1. Update `src/db/schema.sql`
2. Update `src/db/rls.sql` if needed
3. Test with local PostgreSQL
4. Deploy to Supabase

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📞 Support

For issues and questions:
- Check the API documentation at `/docs`
- Review test cases for usage examples
- Check health endpoint for service status
