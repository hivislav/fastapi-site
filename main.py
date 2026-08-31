import os
import json
import urllib.request

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Чат-бот на FastAPI")


# ---------------------------------------------------------------------------
# Простая загрузка .env (без внешней зависимости): читает KEY=VALUE строки
# из файла .env рядом с main.py, не перезаписывая уже заданные переменные.
# ---------------------------------------------------------------------------
def load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

# ---------------------------------------------------------------------------
# Настройки реальной LLM (Yandex Cloud / DeepSeek).
# Значения берутся из переменных окружения (.env) и могут быть переопределены.
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://ai.api.cloud.yandex.net/v1"
)
LLM_MODEL = os.getenv(
    "LLM_MODEL", "gpt://b1gkm5u908if6dc0focb/deepseek-v4-flash/latest"
)
LLM_API_KEY = os.getenv("YANDEX_API_KEY", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))


# ---------------------------------------------------------------------------
# Модель запроса
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# Демо-"нейросеть": отвечает простыми правилами
# ---------------------------------------------------------------------------
def demo_ai(user_text: str) -> str:
    text = user_text.strip().lower()

    greetings = ["привет", "здравствуй", "хай", "добрый день", "hi", "hello"]
    help_words = ["помощь", "help", "что умеешь", "умеешь"]
    how_are_you = ["как дела", "как ты", "what's up"]
    thanks = ["спасибо", "благодарю", "thanks", "thank"]
    bye = ["пока", "до свидания", "прощай", "bye"]

    if any(g in text for g in greetings):
        return "Привет! 👋 Я демо-чат-бот. Спросите меня что-нибудь — я отвечу по правилам, без настоящей нейросети."
    if any(w in text for w in how_are_you):
        return "У меня всё отлично, работает прямо из вашего браузера! А как у вас дела? 😊"
    if any(w in text for w in help_words):
        return "Я умею: здороваться, отвечать на «как дела», благодарить, прощаться. Также я понимаю вопросы, содержащие «бог», «погода» и «имя». Это демо-режим!"
    if any(w in text for w in thanks):
        return "Всегда пожалуйста! Обращайтесь ещё 🙂"
    if any(w in text for w in bye):
        return "До свидания! Буду ждать вас снова 👋"

    if "имя" in text or "зовут" in text:
        return "Меня зовут ДемоБот. Я работаю на FastAPI и отвечаю по строгим правилам 🧠"
    if "бог" in text:
        return "Хм, это глубокий вопрос. В демо-режиме я отвечу философски: каждый сам решает, во что верить. 🙏"
    if "погод" in text:
        return "Я не умею подключаться к реальным сервисам, но могу сказать: на моём сервере всегда солнечно! ☀️"
    if "100" in text and ("лет" in text or "возраст" in text):
        return "Мне всего пару минут — я только что создан на FastAPI! 🎂"

    # Ответ по умолчанию
    return (
        "Интересный вопрос! К сожалению, я демо-бот и знаю только несколько тем "
        "(приветствие, «как дела», помощь, прощание, имя, погода). "
        "Попробуйте спросить «что ты умеешь». 🤖"
    )


# ---------------------------------------------------------------------------
# Вызов реальной LLM через Yandex Cloud (OpenAI-совместимый endпоинт)
# ---------------------------------------------------------------------------
def call_llm(user_text: str) -> str:
    """Отправляет запрос к реальной модели и возвращает текст ответа.

    Использует OpenAI-совместимый формат (chat/completions). Если API-ключ
    не задан или возникла ошибка, возвращает пустую строку — тогда вызывающий
    код может использовать демо-ответ.
    """
    if not LLM_API_KEY:
        return ""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — дружелюбный ассистент на сайте. Отвечай кратко, "
                    "по делу и на том же языке, на котором задан вопрос."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "max_tokens": LLM_MAX_TOKENS,
    }

    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"].get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # Некоторые модели возвращают ответ только в reasoning_content
        reasoning = data["choices"][0]["message"].get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Маршруты
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Главная страница — интерфейс чата."""
    return HTML_CHAT


@app.get("/health")
def health() -> dict:
    """Простой endpoint для проверки состояния сервиса."""
    return {"status": "ok"}


@app.post("/api/chat")
def chat(msg: ChatMessage) -> dict:
    """Принимает сообщение юзера и возвращает ответ бота."""
    if not msg.content.strip():
        return {"user": msg.content, "bot": "Пожалуйста, введите сообщение."}
    answer = call_llm(msg.content)
    if not answer:
        # Если реальная LLM недоступна (нет ключа или ошибка) — демо-режим.
        answer = demo_ai(msg.content)
    return {"user": msg.content, "bot": answer}


# ---------------------------------------------------------------------------
# HTML интерфейс чата
# ---------------------------------------------------------------------------
HTML_CHAT = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Чат-бот</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .layout {
            display: grid;
            grid-template-columns: 260px 1fr;
            gap: 16px;
            height: 100vh;
            max-width: 1200px;
            margin: 0 auto;
            padding: 16px;
        }
        /* ---------- Боковая панель: история юзера ---------- */
        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        .sidebar h2 {
            margin: 0 0 12px;
            font-size: 1rem;
            color: #4a3b8f;
        }
        .user-history {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .history-item {
            background: #f0ebff;
            border-radius: 10px;
            padding: 8px 10px;
            font-size: 0.85rem;
            word-break: break-word;
            border-left: 4px solid #667eea;
        }
        .history-empty {
            color: #999;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 20px;
        }
        /* ---------- Основная область чата ---------- */
        .chat {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        .chat-header {
            padding: 16px 20px;
            background: #5b4bb8;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .chat-header .avatar { font-size: 1.6rem; }
        .chat-header h1 { margin: 0; font-size: 1.2rem; }
        .chat-header .sub { font-size: 0.8rem; opacity: 0.85; }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .msg { display: flex; }
        .msg.user { justify-content: flex-end; }
        .msg.bot { justify-content: flex-start; }
        .bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.4;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .msg.user .bubble {
            background: #667eea;
            color: #fff;
            border-bottom-right-radius: 4px;
        }
        .msg.bot .bubble {
            background: #eee6ff;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        .input-row {
            display: flex;
            gap: 10px;
            padding: 14px;
            border-top: 1px solid #ececec;
            background: #fff;
        }
        .input-row input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 24px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-row input:focus { border-color: #667eea; }
        .input-row button {
            padding: 12px 22px;
            border: none;
            border-radius: 24px;
            background: #667eea;
            color: #fff;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .input-row button:hover { background: #5563c1; }
        .input-row button:disabled { background: #aaa; cursor: not-allowed; }
        .typing {
            color: #888;
            font-size: 0.85rem;
            padding: 4px 16px 0;
        }
        @media (max-width: 768px) {
            .layout { grid-template-columns: 1fr; }
            .sidebar { display: none; }
        }
    </style>
</head>
<body>
    <div class="layout">
        <!-- Боковая панель: история введённых данных юзером -->
        <aside class="sidebar">
            <h2>🕘 История введённых данных</h2>
            <div id="history" class="user-history">
                <div class="history-empty">Здесь появится, что вы вводили</div>
            </div>
        </aside>

        <!-- Основное окно чата -->
        <main class="chat">
            <div class="chat-header">
                <span class="avatar">🤖</span>
                <div>
                    <h1>Чат-бот</h1>
                    <div class="sub">DeepSeek · Yandex Cloud · демо-режим при сбое</div>
                </div>
            </div>
            <div id="messages" class="messages">
                <div class="msg bot">
                    <div class="bubble">Привет! 👋 Я демо-чат-бот. Напишите что-нибудь в поле ниже.</div>
                </div>
            </div>
            <div id="typing" class="typing"></div>
            <div class="input-row">
                <input id="input" type="text" placeholder="Введите сообщение..." autocomplete="off">
                <button id="send">Отправить</button>
            </div>
        </main>
    </div>

    <script>
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const messagesEl = document.getElementById('messages');
        const historyEl = document.getElementById('history');
        const typingEl = document.getElementById('typing');

        function addHistory(text) {
            // убрать заглушку
            const empty = historyEl.querySelector('.history-empty');
            if (empty) empty.remove();
            const div = document.createElement('div');
            div.className = 'history-item';
            div.textContent = text;
            historyEl.prepend(div);
        }

        function addMessage(role, text) {
            const wrap = document.createElement('div');
            wrap.className = 'msg ' + role;
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = text;
            wrap.appendChild(bubble);
            messagesEl.appendChild(wrap);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        async function send() {
            const content = input.value.trim();
            if (!content) return;
            input.value = '';
            sendBtn.disabled = true;

            addMessage('user', content);
            addHistory(content);

            typingEl.textContent = 'ДемоБот печатает...';
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                const data = await res.json();
                addMessage('bot', data.bot);
            } catch (e) {
                addMessage('bot', '⚠️ Ошибка связи с сервером. Попробуйте ещё раз.');
            } finally {
                typingEl.textContent = '';
                sendBtn.disabled = false;
                input.focus();
            }
        }

        sendBtn.addEventListener('click', send);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') send();
        });
        input.focus();
    </script>
</body>
</html>
"""