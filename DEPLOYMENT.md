# 🚀 Deployment Guide

Signal.Engine supports multiple deployment methods: Docker (recommended), Manual, and CI/CD.

This guide provides step-by-step instructions for getting the system up and running in various environments, from local development to production.

---

## 📑 Table of Contents

1. [Docker Deployment (Recommended)](#-docker-deployment-recommended)
2. [Manual Deployment (Local Development)](#️-manual-deployment-local-development)
3. [Cloud Deployment: Vercel + Koyeb](#️-cloud-deployment-vercel--koyeb)
4. [CI/CD Pipeline](#-cicd-pipeline)
5. [Troubleshooting](#-troubleshooting)

---

## 🐳 Docker Deployment (Recommended)

The easiest and most reliable way to run the full stack (Backend + Frontend + Nginx). It ensures consistent environments and avoids dependency conflicts.

### Prerequisites
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Step-by-Step

**1. Configure Environment Variables**
Create a `.env` file in the root directory by copying the example:
```sh
cp .env.example .env
```
Edit `.env` and configure your API keys (e.g., `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) and any necessary database credentials.

**2. Run with Docker Compose**
Start the entire stack in detached mode:
```sh
docker-compose up --build -d
```

**3. Access the Application**
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation (Swagger UI)**: http://localhost:8000/docs

**4. Stop the Stack**
To gracefully shut down all services:
```sh
docker-compose down
```

---

## 🛠️ Manual Deployment (Local Development)

If you need to develop features or debug the application locally without Docker, you can run the backend and frontend separately.

### Backend (Python API)

**Prerequisites:** Python 3.10+ installed.

1. **Install Dependencies:**
   It is recommended to use a virtual environment.
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the API Server:**
   Start the FastAPI application using Uvicorn with hot-reloading enabled.
   ```sh
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend (React App)

**Prerequisites:** Node.js 20+ installed.

1. **Install Dependencies:**
   Navigate to the frontend directory and install the necessary npm packages.
   ```sh
   cd frontend
   npm install
   ```

2. **Run the Development Server:**
   Start Vite's development server.
   ```sh
   npx vite
   ```
   Access the frontend at http://localhost:5173.

---

## ☁️ Cloud Deployment: Vercel + Koyeb

For a production-ready, highly available deployment, we recommend splitting the architecture: hosting the React frontend on **Vercel** and the Python API on **Koyeb**.

### 1. Koyeb (Backend API)

1. **Connect your GitHub Repository** to Koyeb.
2. **Configure Docker Deployment** (Do not use the standard Buildpack):
   - Set the **Dockerfile path** to `Dockerfile.koyeb`.
   - *Note:* This specific Dockerfile utilizes `requirements-deploy.txt`, which drastically reduces the image size (from ~4.5GB to ~300MB) by omitting heavy libraries like `torch` and `gudhi` if they aren't strictly required for the API layer in this configuration.
3. **Configure Environment Variables** in the Koyeb dashboard:
   ```env
   FRONTEND_URL=https://your-app.vercel.app
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-app.vercel.app
   DATA_PROVIDER=yfinance
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ```
   > ⚠️ **Important:** `PORT` is auto-assigned by Koyeb. Do **NOT** set it manually.
4. **Configure Health Checks:** Point the HTTP health check to `GET /api/health`.

### 2. Vercel (Frontend Dashboard)

1. **Connect your GitHub Repository** to Vercel.
2. **Configure Build Settings:**
   - Set **Root Directory** to `frontend/`.
   - **Framework Preset** should be auto-detected as `Vite`.
3. **Configure Environment Variables** in the Vercel dashboard:
   ```env
   VITE_API_URL=https://your-app.koyeb.app
   VITE_API_KEY=your_api_key  # (Only if API_KEY is enforced on Koyeb)
   ```
   > ⚠️ **Important:** `VITE_*` variables are baked in during the build process. **You must redeploy if you change them.**
4. **Routing:** SPA routing is automatically handled via `vercel.json` (routing all traffic to `index.html`).

---

## 🔄 CI/CD Pipeline

The project uses **GitHub Actions** to automate testing and build verification.

### Workflow: `.github/workflows/main.yml`

- **Triggers**: Executes on any `push` or `pull_request` to the `main` branch.
- **Backend Job**: 
  - Sets up Python 3.10.
  - Installs dependencies from `requirements.txt`.
  - Runs the test suite using `pytest` with mocked API credentials to ensure business logic remains intact.
- **Frontend Job**:
  - Sets up Node.js 20.
  - Installs dependencies via `npm ci`.
  - Runs unit tests (`npx vitest run`).
  - Verifies the build process (`npm run build`).

### CI Status Badges
To display the build status on your main repository page, add the following badge to your `README.md`:
```markdown
![CI Status](https://github.com/yourusername/signal-engine/actions/workflows/main.yml/badge.svg)
```

---

## 🛠 Troubleshooting

Common deployment issues and their solutions:

| Symptom | Diagnosis & Fix |
|---------|-----------------|
| **CORS errors in browser console** | The backend is rejecting the frontend's origin. Ensure `FRONTEND_URL` is correctly set on Koyeb to your Vercel URL. |
| **404 error on page refresh (Vercel)** | SPA routing issue. Verify that `frontend/vercel.json` exists and correctly rewrites all routes to `index.html`. |
| **API calls failing/going to localhost** | The frontend is not pointing to the live API. Set `VITE_API_URL` in Vercel environment variables and trigger a **redeploy**. |
| **WebSocket connections rejected** | The backend's allowed origins list is incomplete. Add your Vercel URL to the `CORS_ORIGINS` environment variable on Koyeb. |
| **Koyeb build fails with Out of Memory (OOM)** | The deployment image is too large. Ensure you are using `Dockerfile.koyeb` and `requirements-deploy.txt` (which excludes `torch`/`gudhi`). |
