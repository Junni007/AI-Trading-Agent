# 🦅 Signal.Engine

### Generative Algorithmic Intelligence for the Nifty 500

**Signal.Engine** is an autonomous market analysis system that moves beyond simple price prediction. It uses **Generative AI (Sequence Modeling)** to understand market context and **Reinforcement Learning (PPO)** to execute precision trades.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![React](https://img.shields.io/badge/react-18-cyan.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

---

## 🚀 Key Features

### 🧠 Generative Agent (v3.0)
- **Sequence Modeling (LSTM)**: Processes 50-candle sequences, not just snapshots.
- **Supervised Fine-Tuning (SFT)**: Pre-trained on "Golden Labels" (ZigZag) for common sense.
- **RL Fine-Tuning (PPO)**: Optimized for risk-adjusted returns (Sortino Ratio) in a vectorized environment.

### ⚡ Real-Time Architecture
- **Async Backend**: FastAPI + asyncio for non-blocking analysis.
- **Live Websockets**: Real-time ticker updates and "Thinking" state broadcast.
- **Thread-Safe State**: Robust `threading.Lock` management for concurrent scanning.
- **Global Error Tracing**: Request IDs (`X-Request-ID`) attached to every log and error.

### 🎨 Modern Dashboard
- **React + Vite**: Fast, responsive UI.
- **Live Signal Cards**: See what the agent is "thinking" in real-time.
- **Interactive Charts**: Recharts-based performance visualization.
- **Design System**: Typography-driven, "Industrial Luxury" aesthetic.

---

## 🛠️ Quick Start

### Docker (Recommended)
```bash
# 1. Configure env
cp .env.example .env

# 2. Run
docker-compose up -d
```
Access dashboard at **http://localhost:3000**.

### Manual
See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed manual setup instructions.

---

## 📚 Documentation
- **[Deployment Guide](DEPLOYMENT.md)**: Docker, CI/CD, and Manual setup.
- **[Project Manual](docs/PROJECT_MANUAL.md)**: Architecture deep dive.
- **[Analysis Guide](docs/ANALYSIS_GUIDE.md)**: How to run backtests and analysis.
- **[Data Schemas](docs/DATA_SCHEMAS.md)**: Feature engineering reference.

---

## 🧪 Testing

The project maintains high test coverage:
- **Backend**: `pytest tests/` (50+ tests covering logic, API, and security).
- **Frontend**: `npm run test` (Vitest + React Testing Library).

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
