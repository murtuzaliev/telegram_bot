"""
Полная версия Telegram-бота с исправленным извлечением субтитров YouTube.
Добавлены Inline кнопки, callback-обработчики, улучшенное меню.

Переменные окружения:
  TELEGRAM_TOKEN, OPENROUTER_API_KEY, RENDER_EXTERNAL_URL (webhook), PORT
  USE_POLLING=1 — polling вместо webhook
  YOUTUBE_TRANSCRIPT_PROXY или HTTPS_PROXY — URL прокси (обход блокировки IP)
  GROUP_SPONTANEOUS_CHANCE — шанс % «влезть» в переписку (0–100, по умолч. 15)
  GROUP_COOLDOWN_SEC — пауза между спонтанными ответами (сек., по умолч. 45)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
import time
import urllib.parse
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# YouTube субтитры
try:
    from youtube_transcript_api import YouTubeTranscriptApi

    YT_API_AVAILABLE = True
    YT_API_V1 = not hasattr(YouTubeTranscriptApi, "get_transcript")
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    try:
        from youtube_transcript_api._errors import TooManyRequests
    except ImportError:
        TooManyRequests = None
    try:
        from youtube_transcript_api._errors import IpBlocked, RequestBlocked
    except ImportError:
        IpBlocked = None
        RequestBlocked = None
except ImportError:
    YT_API_AVAILABLE = False
    YT_API_V1 = False
    YouTubeTranscriptApi = None
    TooManyRequests = None
    IpBlocked = None
    RequestBlocked = None

# ----- Логирование -----
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ----- Переменные -----
TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
URL = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
PORT = int(os.getenv("PORT", "8000"))
USE_POLLING = os.getenv("USE_POLLING", "").lower() in ("1", "true", "yes")

# Прокси для YouTube
YOUTUBE_TRANSCRIPT_PROXY = (os.environ.get("YOUTUBE_TRANSCRIPT_PROXY") or "").strip()
HTTPS_PROXY_ENV = (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip().isdigit():
        return default
    return int(str(raw).strip())


GROUP_COOLDOWN_SEC = max(5, _env_int("GROUP_COOLDOWN_SEC", 45))

if not TOKEN:
    logger.error("TELEGRAM_TOKEN не установлен!")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY не установлен!")

# ----- Модели -----
MODELS = {
    "gemini": "🤖 Gemini 2.0 Flash",
    "gpt-4o": "🧠 GPT-4o (Vision)",
    "claude": "🎭 Claude 3.5 Sonnet",
    "gpt-mini": "💚 GPT-4o Mini",
    "llama": "🦙 Llama 3.3 (Free)",
    "deepseek": "🐋 DeepSeek V3",
}

MODEL_IDS = {
    "gemini": "google/gemini-2.0-flash-001",
    "gpt-4o": "openai/gpt-4o",
    "claude": "anthropic/claude-3.5-sonnet",
    "gpt-mini": "openai/gpt-4o-mini",
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek": "deepseek/deepseek-chat",
}

MODELS_INFO = {
    "gemini": {"icon": "🤖", "name": "Gemini 2.0 Flash", "desc": "Быстрая, умная, бесплатно", "vision": True},
    "gpt-4o": {"icon": "🧠", "name": "GPT-4o", "desc": "Видит фото, мощная", "vision": True},
    "claude": {"icon": "🎭", "name": "Claude 3.5 Sonnet", "desc": "Креативная, длинные ответы", "vision": True},
    "gpt-mini": {"icon": "💚", "name": "GPT-4o Mini", "desc": "Экономичная, быстрая", "vision": True},
    "llama": {"icon": "🦙", "name": "Llama 3.3", "desc": "Совсем бесплатно", "vision": False},
    "deepseek": {"icon": "🐋", "name": "DeepSeek V3", "desc": "Мощная, бесплатно", "vision": False},
}

VISION_MODELS = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
]

PERSONAS = {
    "default": {"name": "🤖 Ассистент", "icon": "🤖", "prompt": "Ты полезный и дружелюбный ассистент. Отвечай кратко и по делу."},
    "coder": {"name": "💻 Python Dev", "icon": "💻", "prompt": "Ты Senior Python разработчик. Отвечай с примерами кода, объясняй решения."},
    "translator": {"name": "🌍 Переводчик", "icon": "🌍", "prompt": "Ты профессиональный переводчик. Переводи всё на русский язык."},
    "teacher": {"name": "📚 Учитель", "icon": "📚", "prompt": "Ты учитель английского. Исправляй ошибки, объясняй грамматику."},
    "writer": {"name": "✍️ Копирайтер", "icon": "✍️", "prompt": "Ты креативный копирайтер. Пиши увлекательно, используй метафоры."},
    "psychologist": {"name": "🎭 Психолог", "icon": "🎭", "prompt": "Ты поддерживающий психолог. Задавай уточняющие вопросы, помогай рефлексировать."},
    "smm": {"name": "📱 SMM", "icon": "📱", "prompt": "Ты эксперт по SMM. Генерируй идеи для постов, пиши вовлекающие тексты."},
    "analyst": {"name": "📊 Аналитик", "icon": "📊", "prompt": "Ты бизнес-аналитик. Структурируй информацию, делай выводы."},
}

GROUP_CONFIG = {
    "enabled": True,
    "spontaneous_chance": max(0, min(100, _env_int("GROUP_SPONTANEOUS_CHANCE", 15))),
    "max_context_messages": 8,
    "min_message_length": 12,
}

# Настройки отдельных чатов
group_chat_settings: dict = {}
group_spontaneous_last_reply: dict = defaultdict(float)

user_chats: dict = {}
user_last_message = defaultdict(float)
user_persona: dict = {}
group_messages = defaultdict(list)
MAX_HISTORY_LENGTH = 10

GROUP_CHAT_SYSTEM_PROMPT = (
    "Ты — живой участник русскоязычного Telegram-чата. "
    "Можешь шутить, слегка иронизировать и отвечать с лёгким сарказмом, как с друзьями. "
    "Без мата, оскорблений, травли, политики и токсичности; не переходи на личности и не унижай людей. "
    "Пиши по-русски, коротко: обычно 1–3 предложения, можно эмодзи. "
    "Если тебя упомянули или ответили на твоё сообщение — отвечай по делу, но в том же живом стиле."
)


def get_group_spontaneous_chance(chat_id: int) -> int:
    if chat_id in group_chat_settings and "spontaneous_chance" in group_chat_settings[chat_id]:
        return max(0, min(100, int(group_chat_settings[chat_id]["spontaneous_chance"])))
    return GROUP_CONFIG["spontaneous_chance"]


client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ----- Меню (Reply Keyboard) -----
def get_main_menu() -> ReplyKeyboardMarkup:
    keyboard = [
        ["🤖 Модели", "🎭 Личности"],
        ["🎨 Картинка", "🔗 Анализ ссылки"],
        ["🗑️ Очистить", "ℹ️ Помощь"],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


def get_models_menu() -> ReplyKeyboardMarkup:
    keyboard = [
        ["🤖 Gemini", "🧠 GPT-4o"],
        ["🎭 Claude", "💚 GPT-4o Mini"],
        ["🦙 Llama", "🐋 DeepSeek"],
        ["🔙 Назад"],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


def get_personas_menu() -> ReplyKeyboardMarkup:
    keyboard = [
        ["🤖 Ассистент", "💻 Python Dev"],
        ["🌍 Переводчик", "📚 Учитель"],
        ["✍️ Копирайтер", "🎭 Психолог"],
        ["📱 SMM", "📊 Аналитик"],
        ["🔙 Назад"],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


# ----- Inline меню (кнопки под сообщениями) -----
def get_models_inline() -> InlineKeyboardMarkup:
    """Inline кнопки для выбора моделей"""
    keyboard = [
        [
            InlineKeyboardButton("🤖 Gemini", callback_data="model_gemini"),
            InlineKeyboardButton("🧠 GPT-4o", callback_data="model_gpt-4o"),
        ],
        [
            InlineKeyboardButton("🎭 Claude", callback_data="model_claude"),
            InlineKeyboardButton("💚 GPT-4o Mini", callback_data="model_gpt-mini"),
        ],
        [
            InlineKeyboardButton("🦙 Llama", callback_data="model_llama"),
            InlineKeyboardButton("🐋 DeepSeek", callback_data="model_deepseek"),
        ],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_personas_inline() -> InlineKeyboardMarkup:
    """Inline кнопки для выбора личностей"""
    keyboard = [
        [
            InlineKeyboardButton("🤖 Ассистент", callback_data="persona_default"),
            InlineKeyboardButton("💻 Python Dev", callback_data="persona_coder"),
        ],
        [
            InlineKeyboardButton("🌍 Переводчик", callback_data="persona_translator"),
            InlineKeyboardButton("📚 Учитель", callback_data="persona_teacher"),
        ],
        [
            InlineKeyboardButton("✍️ Копирайтер", callback_data="persona_writer"),
            InlineKeyboardButton("🎭 Психолог", callback_data="persona_psychologist"),
        ],
        [
            InlineKeyboardButton("📱 SMM", callback_data="persona_smm"),
            InlineKeyboardButton("📊 Аналитик", callback_data="persona_analyst"),
        ],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_info_inline() -> InlineKeyboardMarkup:
    """Inline кнопки для информации"""
    keyboard = [
        [InlineKeyboardButton("📊 О боте", callback_data="about_bot")],
        [InlineKeyboardButton("🎭 О личностях", callback_data="about_personas")],
        [InlineKeyboardButton("🤖 О моделях", callback_data="about_models")],
        [InlineKeyboardButton("👥 Групповой режим", callback_data="about_group")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")],
    ]
    return InlineKeyboardMarkup(keyboard)


# ----- AI -----
async def get_ai_response(model_id: str, messages) -> str:
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id,
            messages=messages,
            max_tokens=1500,
            timeout=60.0,
        )
        return response.choices[0].message.content
    except httpx.TimeoutException:
        return "⏰ Превышено время ожидания. Попробуйте позже."
    except Exception as e:
        logger.error("Ошибка API: %s", e)
        return f"❌ Ошибка: {str(e)[:200]}"


# ----- YouTube: извлечение субтитров -----
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/|m\.youtube\.com/watch\?v=)"
    r"([a-zA-Z0-9_-]{11})"
)


def _youtube_proxy_url() -> Optional[str]:
    p = YOUTUBE_TRANSCRIPT_PROXY or HTTPS_PROXY_ENV
    return p if p else None


def _youtube_requests_proxies() -> Optional[dict]:
    p = _youtube_proxy_url()
    if not p:
        return None
    return {"http": p, "https": p}


def _youtube_api_client():
    from youtube_transcript_api.proxies import GenericProxyConfig

    p = _youtube_proxy_url()
    if p:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=p, https_url=p),
        )
    return YouTubeTranscriptApi()


def _is_youtube_blocked_error(text: str) -> bool:
    t = text.lower()
    return (
        "429" in text
        or "too many requests" in t
        or "google.com/sorry" in t
        or "/sorry/index" in t
        or "ipblocked" in t
        or "requestblocked" in t
        or "blocking requests from your ip" in t
    )


def _humanize_youtube_exception(exc: BaseException) -> str:
    if IpBlocked is not None and isinstance(exc, IpBlocked):
        return _proxy_help_message()
    if RequestBlocked is not None and isinstance(exc, RequestBlocked):
        return _proxy_help_message()
    raw = str(exc)
    if _is_youtube_blocked_error(raw):
        return _proxy_help_message()
    if len(raw) > 350:
        return raw[:350] + "…"
    return raw


def _proxy_help_message() -> str:
    return (
        "YouTube *блокирует* запросы с IP вашего сервера (типично для облака: Render, Railway, AWS). "
        "Это *не* значит, что у видео нет субтитров.\n\n"
        "*Решение:* задайте переменную `YOUTUBE_TRANSCRIPT_PROXY` — URL резидентского прокси.\n"
        "Альтернатива: запуск бота на домашнем ПК или VPS с «чистым» IP."
    )


def _extract_youtube_video_id(url: str) -> Optional[str]:
    url = url.strip()
    m = _YOUTUBE_ID_RE.search(url)
    if m:
        return m.group(1)
    if "youtu.be" in url:
        part = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
        return part if len(part) == 11 else None
    return None


def _fetched_to_text(fetched) -> str:
    if fetched is None:
        return ""
    if isinstance(fetched, list) and (not fetched or isinstance(fetched[0], dict)):
        return " ".join(entry.get("text", "") for entry in fetched)
    if hasattr(fetched, "snippets"):
        return " ".join(s.text for s in fetched.snippets)
    return " ".join(getattr(s, "text", str(s)) for s in fetched)


def _extract_youtube_transcript_v1(video_id: str, langs_try: list) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    api = _youtube_api_client()
    if _youtube_proxy_url():
        logger.info("YouTube: используется прокси (v1 API)")

    try:
        fetched = api.fetch(video_id, languages=langs_try)
        text = _fetched_to_text(fetched)
        if text.strip():
            return text[:4000], "auto", None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.warning("fetch: %s", e)
    except Exception as e:
        logger.warning("fetch unexpected: %s", e)
        if IpBlocked is not None and isinstance(e, IpBlocked):
            return None, None, _humanize_youtube_exception(e)
        if _is_youtube_blocked_error(str(e)):
            return None, None, _humanize_youtube_exception(e)

    try:
        tlist = api.list(video_id)
        for t in tlist:
            try:
                data = t.fetch()
                text = _fetched_to_text(data)
                if text.strip():
                    return text[:4000], t.language_code, None
            except Exception as ex:
                if _is_youtube_blocked_error(str(ex)):
                    return None, None, _humanize_youtube_exception(ex)
    except Exception as e:
        logger.exception("list error (v1)")
        return None, None, _humanize_youtube_exception(e)

    return None, None, None


def _extract_youtube_transcript_v0(video_id: str, langs_try: list) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    proxies = _youtube_requests_proxies()
    if proxies:
        logger.info("YouTube: используется прокси (v0 API)")

    kw = {"proxies": proxies} if proxies else {}

    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs_try, **kw)
        text = _fetched_to_text(transcript_data)
        return text[:4000], "auto", None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.warning("get_transcript: %s", e)
    except Exception as e:
        if _is_youtube_blocked_error(str(e)):
            return None, None, _humanize_youtube_exception(e)

    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id, **kw)
        for t in tlist:
            try:
                data = t.fetch()
                text = _fetched_to_text(data)
                if text.strip():
                    return text[:4000], t.language_code, None
            except Exception as ex:
                if _is_youtube_blocked_error(str(ex)):
                    return None, None, _humanize_youtube_exception(ex)
    except Exception as e:
        return None, None, _humanize_youtube_exception(e)

    return None, None, None


def extract_youtube_transcript(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not YT_API_AVAILABLE:
        return None, None, "Библиотека youtube-transcript-api не установлена"

    video_id = _extract_youtube_video_id(url)
    if not video_id:
        return None, None, "Не удалось определить ID видео по ссылке"

    langs_try = ["ru", "uk", "en", "de", "fr", "es", "it", "pt", "pl"]

    if YT_API_V1:
        text, lang, err = _extract_youtube_transcript_v1(video_id, langs_try)
    else:
        text, lang, err = _extract_youtube_transcript_v0(video_id, langs_try)

    if text:
        return text, lang, None
    if err:
        return None, None, err

    return None, None, "Не удалось получить субтитры. Задайте YOUTUBE_TRANSCRIPT_PROXY или проверьте наличие субтитров."


async def extract_text_from_url(url: str) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TelegramBot/1.0)",
            "Accept-Language": "ru,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=20.0, headers=headers, follow_redirects=True) as client_http:
            response = await client_http.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            return text[:4000] if text else None
    except Exception as e:
        logger.error("URL extraction error: %s", e)
        return None


async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    status_msg = await update.message.reply_text("🔍 *Анализирую ссылку...*", parse_mode=ParseMode.MARKDOWN)

    is_youtube = "youtube.com" in url or "youtu.be" in url
    text = None
    language = None
    error_msg = None

    if is_youtube:
        await status_msg.edit_text("🎬 *Обнаружено YouTube видео!*\nПроверяю субтитры...", parse_mode=ParseMode.MARKDOWN)
        text, language, error_msg = await asyncio.to_thread(extract_youtube_transcript, url)
        if error_msg:
            await status_msg.edit_text(f"❌ *Не удалось извлечь субтитры*\n\n{error_msg}", parse_mode=ParseMode.MARKDOWN)
            return
    else:
        text = await extract_text_from_url(url)

    if not text:
        await status_msg.edit_text("❌ *Не удалось извлечь текст из ссылки*", parse_mode=ParseMode.MARKDOWN)
        return

    model_key = context.user_data.get("model", "gemini")
    model_id = MODEL_IDS.get(model_key)

    if is_youtube:
        prompt = f"""
        Транскрипция YouTube видео{f" ({language})" if language else ""}:
        {text}

        Сделай краткое содержание:
        1. О чем видео
        2. Ключевые моменты (3-5 пунктов)
        3. Выводы

        Формат:
        🎬 *О чем видео:*
        [2-3 предложения]
        🔑 *Ключевые моменты:*
        • пункт 1
        • пункт 2
        💡 *Выводы:*
        [1-2 предложения]
        """
    else:
        prompt = f"""
        Сделай краткую выжимку из текста:
        {text}

        Формат:
        📌 *Краткое содержание:*
        [3-5 предложений]
        🔑 *Ключевые моменты:*
        • пункт 1
        • пункт 2
        """

    await status_msg.edit_text("🤔 *Создаю краткое содержание...*", parse_mode=ParseMode.MARKDOWN)

    answer = await get_ai_response(model_id, [{"role": "user", "content": prompt}])

    if answer:
        await status_msg.delete()
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    else:
        await status_msg.edit_text("❌ Не удалось создать краткое содержание.")


# ----- Обработчики меню -----
async def handle_menu_commands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    text = update.message.text
    user_id = update.effective_user.id

    if text == "🤖 Модели":
        await update.message.reply_text(
            "🎯 *Выберите модель:*\n\n"
            "📷 *Vision* — видят фото\n"
            "💎 *Бесплатные*: Gemini, Llama, DeepSeek",
            reply_markup=get_models_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "🎭 Личности":
        await update.message.reply_text(
            "🎭 *Выберите личность:*\n\nКаждая личность меняет стиль общения.",
            reply_markup=get_personas_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "🎨 Картинка":
        await update.message.reply_text(
            "🎨 *Создание картинки*\n\n`/image описание`\n\nПример: `/image кот в космосе`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "🔗 Анализ ссылки":
        await update.message.reply_text(
            "🔗 *Анализ ссылки*\n\nОтправьте ссылку — сделаю краткое содержание.\n\n• YouTube (субтитры)\n• Веб-страницы"
        )
        return True

    if text == "🗑️ Очистить":
        user_chats[user_id] = []
        await update.message.reply_text("🧹 *История диалога очищена!*", reply_markup=get_main_menu(), parse_mode=ParseMode.MARKDOWN)
        return True

    if text == "ℹ️ Помощь":
        await help_command(update, context)
        return True

    if text == "🔙 Назад":
        await update.message.reply_text("👋 *Главное меню*", reply_markup=get_main_menu(), parse_mode=ParseMode.MARKDOWN)
        return True

    # Выбор моделей из Reply меню
    model_map = {"🤖 Gemini": "gemini", "🧠 GPT-4o": "gpt-4o", "🎭 Claude": "claude",
                 "💚 GPT-4o Mini": "gpt-mini", "🦙 Llama": "llama", "🐋 DeepSeek": "deepseek"}
    if text in model_map:
        model_key = model_map[text]
        context.user_data["model"] = model_key
        await update.message.reply_text(
            f"✅ *Модель:* {MODELS_INFO[model_key]['name']}",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    # Выбор личностей из Reply меню
    persona_map = {"🤖 Ассистент": "default", "💻 Python Dev": "coder", "🌍 Переводчик": "translator",
                   "📚 Учитель": "teacher", "✍️ Копирайтер": "writer", "🎭 Психолог": "psychologist",
                   "📱 SMM": "smm", "📊 Аналитик": "analyst"}
    if text in persona_map:
        user_persona[user_id] = persona_map[text]
        await update.message.reply_text(f"✅ *Личность:* {PERSONAS[persona_map[text]]['name']}", reply_markup=get_main_menu(), parse_mode=ParseMode.MARKDOWN)
        return True

    return False


# ----- Inline Callback обработчик -----
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    # Выбор модели
    if query.data.startswith("model_"):
        model_key = query.data.replace("model_", "")
        context.user_data["model"] = model_key
        await query.edit_message_text(
            f"✅ *Установлена модель:* {MODELS_INFO[model_key]['name']}\n\n{MODELS_INFO[model_key]['desc']}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Выбор личности
    if query.data.startswith("persona_"):
        persona_key = query.data.replace("persona_", "")
        user_persona[user_id] = persona_key
        await query.edit_message_text(
            f"✅ *Установлена личность:* {PERSONAS[persona_key]['name']}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Информационные разделы
    if query.data == "about_bot":
        await query.edit_message_text(
            "🤖 *О боте*\n\nВерсия 2.0\n"
            "• 6 моделей ИИ\n• 8 личностей\n• YouTube анализ\n• Генерация картинок\n• Голосовые сообщения",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_info_inline(),
        )
        return

    if query.data == "about_personas":
        text = "🎭 *Личности*\n\n"
        for key, val in PERSONAS.items():
            text += f"{val['icon']} *{val['name']}* — {val['prompt'][:50]}...\n\n"
        await query.edit_message_text(text[:4000], parse_mode=ParseMode.MARKDOWN, reply_markup=get_info_inline())
        return

    if query.data == "about_models":
        text = "🤖 *Модели*\n\n"
        for key, val in MODELS_INFO.items():
            vision = "📷 Vision" if val["vision"] else "📝 Только текст"
            text += f"{val['icon']} *{val['name']}* — {val['desc']} ({vision})\n\n"
        await query.edit_message_text(text[:4000], parse_mode=ParseMode.MARKDOWN, reply_markup=get_info_inline())
        return

    if query.data == "about_group":
        chance = GROUP_CONFIG["spontaneous_chance"]
        await query.edit_message_text(
            f"👥 *Групповой режим*\n\n"
            f"• Упоминание @бота — всегда отвечаю\n"
            f"• Ответ на моё сообщение — всегда отвечаю\n"
            f"• Спонтанный ответ: {chance}% шанс\n"
            f"• Антиспам: {GROUP_COOLDOWN_SEC} сек\n\n"
            f"*Админы:* `/groupchance 20` — изменить шанс",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_info_inline(),
        )
        return

    if query.data == "back_to_menu":
        await query.edit_message_text(
            "👋 *Главное меню*\n\nВыберите действие:",
            reply_markup=get_models_inline(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return


# ----- Команды -----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    model_key = context.user_data.get("model", "gemini")
    persona_key = user_persona.get(user_id, "default")
    bot_username = (await context.bot.get_me()).username

    await update.message.reply_text(
        f"👋 *Привет! Я твой AI-ассистент!*\n\n"
        f"⚙️ **Модель:** {MODELS_INFO[model_key]['name']}\n"
        f"🎭 **Личность:** {PERSONAS[persona_key]['name']}\n\n"
        f"Я умею:\n"
        f"📝 Отвечать на вопросы\n"
        f"🎨 Генерировать картинки\n"
        f"🔗 Анализировать ссылки (включая YouTube!)\n"
        f"📷 Распознавать фото\n"
        f"🎙️ Слышать голос\n"
        f"👥 *В группах:* упомяните @{bot_username} или ответьте на моё сообщение\n\n"
        f"👇 *Используй меню внизу для навигации!*",
        reply_markup=get_main_menu(),
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_username = (await context.bot.get_me()).username
    await update.message.reply_text(
        f"📚 *Помощь*\n\n"
        f"/start — Главное меню\n"
        f"/help — Это сообщение\n"
        f"/image [описание] — Создать картинку\n\n"
        f"*В группах:*\n"
        f"• Упомяните @{bot_username}\n"
        f"• Ответьте на моё сообщение\n"
        f"• Админы: `/groupchance 0-100`\n\n"
        f"*YouTube:* если не работают субтитры, добавьте переменную YOUTUBE_TRANSCRIPT_PROXY",
        parse_mode=ParseMode.MARKDOWN,
    )


async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text(
            "❌ *Ошибка:* напишите описание\n\nПример: `/image кот в космосе`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    prompt = " ".join(context.args)
    status_msg = await update.message.reply_text(f"⏳ *Генерирую:* `{prompt}`...", parse
