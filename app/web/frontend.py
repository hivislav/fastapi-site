"""Доступ к фронтенду чата.

HTML-интерфейс вынесен в отдельный файл chat.html, чтобы верстать его можно
было независимо от Python-кода. Этот модуль отдаёт путь к файлу.
"""

import os

CHAT_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")