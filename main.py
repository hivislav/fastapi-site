from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Мой сайт на FastAPI")


HTML_BASE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Добро пожаловать</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
        }
        .card {
            text-align: center;
            background: rgba(255, 255, 255, 0.12);
            padding: 48px 56px;
            border-radius: 20px;
            backdrop-filter: blur(8px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
        }
        h1 { margin: 0 0 12px; font-size: 2.2rem; }
        p { margin: 0; font-size: 1.1rem; opacity: 0.9; }
        code {
            background: rgba(0, 0, 0, 0.25);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Привет, мир! 👋</h1>
        <p>Ваш сайт на <strong>FastAPI</strong> запущен и работает.</p>
        <p style="margin-top:16px;">Сервер слушает адрес <code>127.0.0.1:8000</code></p>
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Главная страница сайта."""
    return HTML_BASE


@app.get("/health")
def health() -> dict:
    """Простой endpoint для проверки состояния сервиса."""
    return {"status": "ok"}