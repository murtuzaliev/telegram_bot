"""
Полная версия Telegram-бота с исправленным извлечением субтитров YouTube.
Переменные окружения: TELEGRAM_TOKEN, OPENROUTER_API_KEY, RENDER_EXTERNAL_URL (для webhook), PORT.
Опционально: USE_POLLING=1 — локальный запуск через polling вместо webhook.
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
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# YouTube субтитры
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TooManyRequests,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    YT_API_AVAILABLE = True
except ImportError:
    YT_API_AVAILABLE = False

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
    "spontaneous_chance": 15,
    "max_context_messages": 8,
    "min_message_length": 20,
}

user_chats: dict = {}
user_last_message = defaultdict(float)
user_persona: dict = {}
group_messages = defaultdict(list)
MAX_HISTORY_LENGTH = 10

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ----- Меню -----
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


# ----- YouTube: исправленное извлечение -----
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/|m\.youtube\.com/watch\?v=)"
    r"([a-zA-Z0-9_-]{11})"
)


def _extract_youtube_video_id(url: str) -> Optional[str]:
    url = url.strip()
    m = _YOUTUBE_ID_RE.search(url)
    if m:
        return m.group(1)
    if "youtu.be" in url:
        part = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
        return part if len(part) == 11 else None
    if "v=" in url:
        part = url.split("v=")[1].split("&")[0]
        return part if len(part) == 11 else None
    return None


def extract_youtube_transcript(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Возвращает (text, language_code_or_label, error_message).
    """
    if not YT_API_AVAILABLE:
        return None, None, "Библиотека youtube-transcript-api не установлена"

    video_id = _extract_youtube_video_id(url)
    if not video_id:
        return None, None, "Не удалось определить ID видео по ссылке"

    langs_try = [
        "ru",
        "uk",
        "en",
        "de",
        "fr",
        "es",
        "it",
        "pt",
        "pl",
        "ja",
        "ko",
        "zh-Hans",
        "zh-Hant",
        "hi",
        "tr",
    ]

    # 1) get_transcript — перебор языков по порядку
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs_try)
        text = " ".join(entry["text"] for entry in transcript_data)
        return text[:4000], "auto", None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.warning("get_transcript: %s", e)
    except TooManyRequests as e:
        logger.warning("TooManyRequests: %s", e)
        return None, None, "Слишком много запросов к YouTube. Попробуйте позже."
    except Exception as e:
        logger.warning("get_transcript unexpected: %s", e)

    # 2) Любая дорожка: сначала ручные, потом авто
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        manual = [t for t in tlist if not t.is_generated]
        generated = [t for t in tlist if t.is_generated]
        for t in manual + generated:
            try:
                data = t.fetch()
                text = " ".join(entry["text"] for entry in data)
                return text[:4000], t.language_code, None
            except Exception as ex:
                logger.debug("fetch failed %s: %s", getattr(t, "language_code", "?"), ex)

        # 3) Перевод EN -> RU
        try:
            tr = tlist.find_transcript(["en"])
            data = tr.translate("ru").fetch()
            text = " ".join(entry["text"] for entry in data)
            return text[:4000], "ru (перевод с en)", None
        except Exception as ex:
            logger.warning("translate fallback: %s", ex)

    except TranscriptsDisabled:
        return None, None, "У этого видео отключены субтитры."
    except (NoTranscriptFound, VideoUnavailable) as e:
        return None, None, f"Субтитры недоступны: {e}"
    except TooManyRequests:
        return None, None, "Слишком много запросов к YouTube. Попробуйте позже."
    except Exception as e:
        logger.exception("list_transcripts error")
        return None, None, f"Ошибка: {str(e)[:200]}"

    return (
        None,
        None,
        "Не удалось получить субтитры. Возможно, их нет, видео недоступно или YouTube блокирует запрос с сервера.",
    )


async def extract_text_from_url(url: str) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TelegramBot/1.0; +https://telegram.org)",
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

    text = None
    is_youtube = "youtube.com" in url or "youtu.be" in url
    language = None
    error_msg = None

    if is_youtube:
        await status_msg.edit_text(
            "🎬 *Обнаружено YouTube видео!*\nПроверяю субтитры (в т.ч. авто)...",
            parse_mode=ParseMode.MARKDOWN,
        )
        text, language, error_msg = extract_youtube_transcript(url)

        if error_msg:
            await status_msg.edit_text(
                f"❌ *Не удалось извлечь субтитры*\n\n"
                f"{error_msg}\n\n"
                f"💡 *Совет:* попробуйте другое видео с субтитрами или проверьте ссылку.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
    else:
        text = await extract_text_from_url(url)

    if not text and not is_youtube:
        await status_msg.edit_text(
            "❌ *Не удалось извлечь текст из ссылки*\n\n"
            "Проверьте, что страница открывается и содержит текст (не только JS).",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    model_key = context.user_data.get("model", "gemini")
    model_id = MODEL_IDS.get(model_key)

    if is_youtube:
        lang_text = f" ({language})" if language else ""
        prompt = f"""
        Это транскрипция (субтитры) YouTube видео{lang_text}.

        Сделай краткое содержание видео:
        1. О чем видео (основная тема)
        2. Ключевые моменты (3-5 пунктов)
        3. Главные выводы

        Транскрипция:
        {text}

        Ответь на русском языке в формате:
        🎬 *О чем видео:*
        [2-3 предложения]

        🔑 *Ключевые моменты:*
        • [пункт 1]
        • [пункт 2]
        • [пункт 3]

        💡 *Выводы:*
        [1-2 предложения]
        """
    else:
        prompt = f"""
        Сделай краткую выжимку из следующего текста. Выдели главные мысли и ключевые факты.

        Текст:
        {text}

        Ответь в формате:
        📌 *Краткое содержание:*
        [3-5 предложений]

        🔑 *Ключевые моменты:*
        • [пункт 1]
        • [пункт 2]
        • [пункт 3]
        """

    await status_msg.edit_text("🤔 *Создаю краткое содержание...*", parse_mode=ParseMode.MARKDOWN)

    answer = await get_ai_response(model_id, [{"role": "user", "content": prompt}])

    if answer:
        await status_msg.delete()
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    else:
        await status_msg.edit_text("❌ Не удалось создать краткое содержание. Попробуйте позже.")


# ----- Меню -----
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
            "🎭 *Выберите личность:*\n\n"
            "Каждая личность меняет стиль общения и подход к ответам.",
            reply_markup=get_personas_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "🎨 Картинка":
        await update.message.reply_text(
            "🎨 *Создание картинки*\n\n"
            "Просто напишите: `/image описание`\n\n"
            "Примеры:\n"
            "`/image кот в космосе`\n"
            "`/image логотип минимализм`\n"
            "`/image закат над морем, цифровой арт`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "🔗 Анализ ссылки":
        await update.message.reply_text(
            "🔗 *Анализ ссылки*\n\n"
            "Отправьте ссылку — сделаю краткое содержание.\n\n"
            "• YouTube (субтитры, в т.ч. авто)\n"
            "• Статьи и страницы с текстом\n\n"
            "*Важно:* если на сервере YouTube блокирует запросы, субтитры могут не скачаться."
        )
        return True

    if text == "🗑️ Очистить":
        user_chats[user_id] = []
        await update.message.reply_text(
            "🧹 *История диалога очищена!*",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    if text == "ℹ️ Помощь":
        await help_command(update, context)
        return True

    if text == "🔙 Назад":
        await update.message.reply_text(
            "👋 *Главное меню*",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    model_map = {
        "🤖 Gemini": "gemini",
        "🧠 GPT-4o": "gpt-4o",
        "🎭 Claude": "claude",
        "💚 GPT-4o Mini": "gpt-mini",
        "🦙 Llama": "llama",
        "🐋 DeepSeek": "deepseek",
    }
    if text in model_map:
        model_key = model_map[text]
        context.user_data["model"] = model_key
        model_name = MODELS_INFO[model_key]["name"]
        await update.message.reply_text(
            f"✅ *Установлена модель:* {model_name}\n\n{MODELS_INFO[model_key]['desc']}",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    persona_map = {
        "🤖 Ассистент": "default",
        "💻 Python Dev": "coder",
        "🌍 Переводчик": "translator",
        "📚 Учитель": "teacher",
        "✍️ Копирайтер": "writer",
        "🎭 Психолог": "psychologist",
        "📱 SMM": "smm",
        "📊 Аналитик": "analyst",
    }
    if text in persona_map:
        persona_key = persona_map[text]
        user_persona[user_id] = persona_key
        persona_name = PERSONAS[persona_key]["name"]
        await update.message.reply_text(
            f"✅ *Установлена личность:* {persona_name}",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN,
        )
        return True

    return False


# ----- Команды -----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    model_key = context.user_data.get("model", "gemini")
    current_model = MODELS_INFO[model_key]["name"]
    current_persona = user_persona.get(user_id, "default")
    persona_name = PERSONAS[current_persona]["name"]
    bot_username = (await context.bot.get_me()).username

    await update.message.reply_text(
        f"👋 *Привет! Я твой AI-ассистент!*\n\n"
        f"⚙️ **Модель:** {current_model}\n"
        f"🎭 **Личность:** {persona_name}\n\n"
        f"Я умею:\n"
        f"📝 Отвечать на вопросы\n"
        f"🎨 Генерировать картинки\n"
        f"🔗 Анализировать ссылки (включая YouTube!)\n"
        f"📷 Распознавать фото\n"
        f"🎙️ Слышать голос\n"
        f"👥 *В группах:* упомяните меня @{bot_username} или ответьте на моё сообщение\n\n"
        f"👇 *Используй меню внизу для навигации!*",
        reply_markup=get_main_menu(),
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_username = (await context.bot.get_me()).username
    help_text = f"""
📚 *Доступные команды:*

/start - Главное меню
/help - Показать это сообщение
/image [описание] - Создать картинку

*Как пользоваться:*
• Используйте меню внизу для навигации
• Просто отправьте текст - я отвечу
• Отправьте фото с вопросом - я проанализирую
• Отправьте голосовое сообщение - я распознаю
• **Отправьте ссылку** — краткое содержание (страница или YouTube)

*В группах:*
• Упомяните меня @{bot_username}
• Ответьте на моё сообщение
• Иногда отвечаю сам (~{GROUP_CONFIG['spontaneous_chance']}% шанс)

*YouTube:* субтитры (ручные и авто), при необходимости перевод EN→RU
    """
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text(
            "❌ *Ошибка:* напишите описание после команды.\n\n"
            "Пример:\n`/image красный закат над морем`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    prompt = " ".join(context.args)
    status_msg = await update.message.reply_text(
        f"⏳ *Генерирую:* `{prompt}`...",
        parse_mode=ParseMode.MARKDOWN,
    )
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&width=1024&height=1024"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client_http:
            response = await client_http.get(image_url)
            if response.status_code == 200:
                await update.message.reply_photo(
                    photo=response.content,
                    caption=f"🎨 *Результат:* `{prompt}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            else:
                await status_msg.edit_text("❌ Сервис генерации временно недоступен.")
    except Exception as e:
        logger.error("Image Gen Error: %s", e)
        await status_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")


# ----- Личные сообщения -----
async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if await handle_menu_commands(update, context):
        return

    current_time = time.time()
    if current_time - user_last_message[user_id] < 1:
        await update.message.reply_text("⏳ Подождите секунду...")
        return
    user_last_message[user_id] = current_time

    if update.message.text and re.match(r"https?://[^\s]+", update.message.text):
        url_match = re.search(r"https?://[^\s]+", update.message.text)
        if url_match:
            await summarize_url(update, context, url_match.group())
            return

    if user_id not in user_chats:
        user_chats[user_id] = []

    model_key = context.user_data.get("model", "gemini")
    model_id = MODEL_IDS.get(model_key)
    persona_key = user_persona.get(user_id, "default")
    persona_prompt = PERSONAS[persona_key]["prompt"]

    content = []

    if update.message.text:
        content = [{"type": "text", "text": update.message.text}]

    elif update.message.photo:
        if model_id not in VISION_MODELS:
            await update.message.reply_text(
                f"❌ Модель {MODELS_INFO[model_key]['name']} не умеет анализировать фото.\n"
                "Выберите Gemini или GPT-4o в меню «Модели»."
            )
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        caption = update.message.caption or "Что на этой картинке?"
        content = [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]

    elif update.message.voice:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        voice_file = await update.message.voice.get_file()
        file_path = f"voice_{user_id}_{int(time.time())}.ogg"

        try:
            await voice_file.download_to_drive(file_path)
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            await update.message.reply_text(
                f"🎤 *Распознано:* {transcript.text}",
                parse_mode=ParseMode.MARKDOWN,
            )
            content = [{"type": "text", "text": transcript.text}]
        except Exception as e:
            logger.error("Whisper Error: %s", e)
            await update.message.reply_text("❌ Ошибка распознавания голоса.")
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    if not content:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    user_chats[user_id].append({"role": "user", "content": content})

    messages_for_api = [{"role": "system", "content": persona_prompt}]
    for m in user_chats[user_id][-MAX_HISTORY_LENGTH:]:
        if isinstance(m["content"], list):
            text_parts = [item["text"] for item in m["content"] if item.get("type") == "text"]
            combined_text = " ".join(text_parts) if text_parts else ""
            messages_for_api.append({"role": m["role"], "content": combined_text})
        else:
            messages_for_api.append(m)

    if messages_for_api:
        messages_for_api[-1]["content"] = content

    answer = await get_ai_response(model_id, messages_for_api)
    if not answer:
        answer = "⚠️ Ошибка связи с AI. Попробуйте позже."

    user_chats[user_id].append({"role": "assistant", "content": answer})

    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i : i + 4096])
    else:
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)


# ----- Группы -----
async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    chat_id = message.chat_id

    if message.from_user and message.from_user.is_bot:
        return

    user_name = message.from_user.first_name if message.from_user else "Пользователь"
    text = message.text or message.caption or ""

    group_messages[chat_id].append(
        {
            "name": user_name,
            "text": text,
            "user_id": message.from_user.id if message.from_user else 0,
            "timestamp": time.time(),
        }
    )
    if len(group_messages[chat_id]) > 50:
        group_messages[chat_id] = group_messages[chat_id][-50:]

    bot_info = await context.bot.get_me()
    bot_username = bot_info.username
    bot_id = bot_info.id

    should_reply = False
    reply_type = "none"

    if message.text and f"@{bot_username}" in message.text:
        should_reply = True
        reply_type = "mention"
    elif (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    ):
        should_reply = True
        reply_type = "reply"
    elif (
        len(text) >= GROUP_CONFIG["min_message_length"]
        and random.randint(1, 100) <= GROUP_CONFIG["spontaneous_chance"]
    ):
        should_reply = True
        reply_type = "spontaneous"

    if not should_reply:
        return

    context_lines = []
    for msg in group_messages[chat_id][-GROUP_CONFIG["max_context_messages"] :]:
        if msg["text"]:
            context_lines.append(f"{msg['name']}: {msg['text']}")

    if not context_lines:
        return

    context_text = "\n".join(context_lines)

    prompts = {
        "mention": f"""Ты участник чата в Telegram. Тебя упомянули.
Ответь весело и коротко (1-2 предложения), можно с эмодзи.

Контекст:
{context_text}

Твой ответ (только ответ):""",
        "reply": f"""Ты участник чата. Тебе ответили.
Будь дружелюбным, коротко (1-2 предложения).

Контекст:
{context_text}

Твой ответ:""",
        "spontaneous": f"""Ты участник чата. Люди что-то обсуждают.
Напиши короткий комментарий, шутку или вопрос по теме (1-2 предложения).

Контекст:
{context_text}

Твой ответ:""",
    }

    prompt = prompts.get(reply_type, prompts["mention"])
    model_key = context.user_data.get("model", "gemini") if context.user_data else "gemini"
    model_id = MODEL_IDS.get(model_key, MODEL_IDS["gemini"])
    user_id = message.from_user.id if message.from_user else 0
    persona_key = user_persona.get(user_id, "default")
    persona_prompt = PERSONAS[persona_key]["prompt"]

    answer = await get_ai_response(
        model_id,
        [{"role": "system", "content": persona_prompt}, {"role": "user", "content": prompt}],
    )
    if answer:
        await message.reply_text(answer[:4096])


# ----- Сборка Application -----
def build_application() -> Application:
    app = (
        Application.builder()
        .token(TOKEN)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("image", generate_image_command))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, handle_private_message))
    app.add_handler(MessageHandler(filters.ChatType.GROUPS & ~filters.COMMAND, handle_group_message))
    return app


application = build_application()


async def telegram_webhook(request: Request) -> Response:
    if request.method != "POST":
        return PlainTextResponse("Use POST", status_code=405)
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return Response()


async def health(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@asynccontextmanager
async def lifespan(app: Starlette):
    await application.initialize()
    await application.start()
    if URL and not USE_POLLING:
        webhook_full = f"{URL}/webhook"
        await application.bot.set_webhook(url=webhook_full, allowed_updates=Update.ALL_TYPES)
        logger.info("Webhook set to %s", webhook_full)
    yield
    if URL and not USE_POLLING:
        await application.bot.delete_webhook(drop_pending_updates=False)
    await application.stop()
    await application.shutdown()


starlette_app = Starlette(
    routes=[
        Route("/webhook", telegram_webhook, methods=["POST"]),
        Route("/", health, methods=["GET"]),
    ],
    lifespan=lifespan,
)


def main() -> None:
    if USE_POLLING or not URL:
        logger.info("Запуск polling (USE_POLLING или нет RENDER_EXTERNAL_URL)")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        import uvicorn

        logger.info("Запуск uvicorn на порту %s", PORT)
        uvicorn.run(starlette_app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
