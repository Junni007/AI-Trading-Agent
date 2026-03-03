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

## ☁️ Vercel (Frontend) + Koyeb (Backend)

Split deployment: React frontend on Vercel, Python API on Koyeb.

### Koyeb (Backend)

1. **Connect your GitHub repo** to Koyeb.
2. **Set Root Directory** to `/` (project root).
3. **Set Environment Variables** on Koyeb:
   ```
   FRONTEND_URL=https://your-app.vercel.app
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   DATA_PROVIDER=yfinance
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ```
   > `PORT` is auto-assigned by Koyeb — do NOT set it manually.
4. **Deployment** uses the `Procfile` and `runtime.txt` automatically.
5. **Health Check**: Set to `GET /api/health`.

### Vercel (Frontend)

1. **Connect your GitHub repo** to Vercel.
2. **Set Root Directory** to `frontend/`.
3. **Framework Preset**: Vite (auto-detected).
4. **Set Environment Variables** on Vercel:
   ```
   VITE_API_URL=https://your-app.koyeb.app
   VITE_API_KEY=your_api_key (if API_KEY is set on Koyeb)
   ```
   > `VITE_*` vars are baked at build time. **Redeploy after changing them.**
5. **SPA Routing**: Handled by `vercel.json` (all routes → `index.html`).

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| CORS errors in browser console | Set `FRONTEND_URL` on Koyeb to your Vercel URL |
| 404 on page refresh | Verify `frontend/vercel.json` exists with rewrites |
| API calls go to localhost | Set `VITE_API_URL` in Vercel env vars & redeploy |
| WebSocket rejected | Add your Vercel URL to `CORS_ORIGINS` on Koyeb |
| Koyeb build OOM | Use `requirements-deploy.txt` without torch/gudhi |

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
