# Netflix RAG Agent

This repo contains a lightweight Retrieval-Augmented Generation demo built with FastAPI, Azure OpenAI, Azure AI Search, and a static Tailwind UI.

## Local Dev (existing workflow)

The original workflow still works:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000

cd ../web
python -m http.server 8080 --bind 0.0.0.0
```

Ensure `backend/.env` contains your real Azure credentials (never commit those!) and browse to `http://localhost:8080/index.html`.

## Dockerized Setup

1. Copy the sample env and fill in your Azure values:
   ```bash
   cp backend/.env.example backend/.env
   # edit backend/.env with real keys
   ```
   - For vector search, also configure `AZURE_OPENAI_EMBED_*`, `AZURE_SEARCH_VECTOR_FIELD`, and update `AZURE_SEARCH_API_VERSION` to a vector-enabled API version such as `2024-07-01-preview` (or the preview supported in your region).
2. Build and run the stack:
   ```bash
   docker compose up --build
   ```
   This starts the FastAPI backend on `http://localhost:8000` and an nginx-hosted web UI on `http://localhost:8080`.
3. Stop everything with `docker compose down`.

### Hot reload tips

- For quick iterations on the backend, rebuild after code changes: `docker compose build backend`.
- For front-end tweaks, rebuild `docker compose build web` or copy the file into the container manually.

## LAN / Mobile testing

If you want a phone on the same Wi‑Fi to use the app, ensure Docker is running and browse to `http://<your-mac-ip>:8080/index.html`. The front end automatically proxies requests to `<your-mac-ip>:8000`.

## Housekeeping

- `.dockerignore` protects secrets by excluding `.env` files from images.
- Rotate any leaked Azure keys immediately and avoid committing real credentials.
