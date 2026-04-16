# Recuperacao Judicial Copilot MVP

Minimal FastAPI prototype for a Brazilian law firm focused on `recuperacao judicial`.

## What it does

- Chat-style legal copilot interface
- Upload `PDF` or `TXT`
- Uses the uploaded document text in the AI response when relevant
- Shows document status and allows clearing the loaded document

## Stack

- Backend: FastAPI
- Frontend: vanilla HTML, CSS, JavaScript
- LLM: OpenAI API
- PDF parsing: PyPDF2

## Local run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Or create a local `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Optional model override:

```powershell
$env:OPENAI_MODEL="gpt-4.1-mini"
```

4. Start the app:

```bash
uvicorn app:app --reload
```

5. Open:

```text
http://127.0.0.1:8000
```

You can either:

- set `OPENAI_API_KEY` in the server environment, or
- paste an API key into the `OpenAI API key` field in the UI for demo use

## Deployment

### Render

This repo includes `render.yaml`, so Render can deploy it as a web service.

Where to insert API key on Render:

- In the service environment variables, set `OPENAI_API_KEY`
- Optionally set `OPENAI_MODEL`

If you do not want to store a server-side key, the UI also accepts a temporary API key for demo sessions.

## One-click Render setup

After this repo is pushed to GitHub, you can deploy it with Render Blueprints using the repo's `render.yaml`.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/felipemabraga/pose-estimation-ponderada)

### Notes

- Document storage is in-memory only
- This is a demo, not a production legal system
- Outputs should always be validated by a lawyer
