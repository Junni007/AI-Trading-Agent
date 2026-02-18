# 🚀 Deployment Guide

Signal Engine supports multiple deployment methods: Docker (recommended), Manual, and CI/CD.

---

## 🐳 Docker Deployment (Recommended)

The easiest way to run the full stack (Backend + Frontend + Nginx).

### Prerequisites
- Docker Engine & Docker Compose installed

### 1. Configure Environment
Create a `.env` file in the root directory:
```bash
cp .env.example .env
# Edit .env and set your ALPACA_API_KEY, ALPACA_SECRET_KEY, etc.
```

### 2. Run with Compose
```bash
docker-compose up --build -d
```

### 3. Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. Stop
```bash
docker-compose down
```

---

## 🛠️ Manual Deployment

### Backend
1. **Python 3.10+** required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend
1. **Node.js 20+** required.
2. Install & Run:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Access at http://localhost:5173

---

## 🔄 CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration.

### Workflow: `.github/workflows/main.yml`
- **Triggers**: Push or PR to `main` branch.
- **Backend Job**: 
  - Sets up Python 3.10
  - Installs requirements
  - Runs `pytest` with mocked credentials
- **Frontend Job**:
  - Sets up Node 20
  - Installs dependencies
  - Runs tests (`npm run test:run`)
  - Builds application (`npm run build`)

### Status Badges
You can add a workflow status badge to your README:
`![CI Status](https://github.com/yourusername/signal-engine/actions/workflows/main.yml/badge.svg)`
