import os
import asyncio
import logging
import base64
import urllib.parse
import time
import re
import random
from collections import defaultdict
import httpx
from starlette.applications import Starlette
from starlette.responses import Response, PlainTextResponse
from starlette.requests import Request
from starlette.routing import Route
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from telegram.constants import ParseMode

from openai import OpenAI
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ПЕРЕМЕННЫЕ =====
TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.getenv("PORT", 8000))

# ===== НАСТРОЙКИ МОДЕЛЕЙ =====
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
    "claude": {"icon": "🎭", "name": "Claude 3.5 Sonnet", "desc": "Креативная", "vision": True},
    "gpt-mini": {"icon": "💚", "name": "GPT-4o Mini", "desc": "Экономичная", "vision": True},
    "llama": {"icon": "🦙", "name": "Llama 3.3", "desc": "Бесплатно", "vision": False},
    "deepseek": {"icon": "🐋", "name": "DeepSeek V3", "desc": "Мощная", "vision": False},
}

VISION_MODELS = ["google/gemini-2.0-flash-001", "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]

# ===== СИСТЕМА ЛИЧНОСТЕЙ =====
PERSONAS = {
    "default": {"name": "🤖 Ассистент", "prompt": "Ты полезный ассистент. Отвечай кратко."},
    "coder": {"name": "💻 Python Dev", "prompt": "Ты Senior Python разработчик. Давай код и объяснения."},
    "translator": {"name": "🌍 Переводчик", "prompt": "Ты переводчик. Переводи всё на русский."},
    "analyst": {"name": "📊 Аналитик", "prompt": "Ты бизнес-аналитик. Структурируй информацию."}
}

# Хранилища
user_chats = {}
user_persona = {}
MAX_HISTORY_LENGTH = 10

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ===== МЕНЮ =====

def get_main_menu():
    keyboard = [["🤖 Модели", "🎭 Личности"], ["🎨 Картинка", "🔗 Анализ ссылки"], ["🗑️ Очистить", "ℹ️ Помощь"]]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# ===== ФУНКЦИИ ИЗВЛЕЧЕНИЯ ДАННЫХ =====

def extract_youtube_transcript(url: str) -> tuple:
    """Извлечение субтитров с ПРИНУДИТЕЛЬНЫМ переводом на русский"""
    try:
        video_id = None
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "shorts/" in url:
                video_id = url.split("shorts/")[1].split("?")[0]
        
        if not video_id: return None, None
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # Пытаемся найти русский (оригинал или авто)
            transcript = transcript_list.find_transcript(['ru'])
            language = "русский"
        except:
            try:
                # Пытаемся найти английский и ПЕРЕВЕСТИ его на русский
                raw_transcript = transcript_list.find_transcript(['en'])
                transcript = raw_transcript.translate('ru')
                language = "английский (авто-перевод)"
            except:
                try:
                    # Берем вообще любой доступный язык и переводим на русский
                    any_lang = list(transcript_list._manually_created_transcripts.keys()) + \
                               list(transcript_list._generated_transcripts.keys())
                    raw_transcript = transcript_list.find_transcript([any_lang[0]])
                    transcript = raw_transcript.translate('ru')
                    language = f"авто-перевод с {any_lang[0]}"
                except:
                    return None, None

        data = transcript.fetch()
        text = " ".join([t['text'] for t in data])
        return text[:6000], language
    except Exception as e:
        logger.error(f"YouTube Error: {e}")
        return None, None

async def extract_text_from_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0) as h_client:
            resp = await h_client.get(url, follow_redirects=True)
            soup = BeautifulSoup(resp.content, 'html.parser')
            for s in soup(["script", "style", "nav", "footer"]): s.decompose()
            return soup.get_text()[:4000]
    except: return None

# ===== ЛОГИКА ИИ =====

async def get_ai_response(model_id, messages):
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id, messages=messages, max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Ошибка ИИ: {str(e)[:100]}"

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    status_msg = await update.message.reply_text("🔍 Анализирую...")
    is_yt = "youtube.com" in url or "youtu.be" in url
    
    if is_yt:
        text, lang = extract_youtube_transcript(url)
    else:
        text, lang = await extract_text_from_url(url), "веб-страница"

    if not text:
        await status_msg.edit_text("❌ Не удалось получить текст. Возможно, нет субтитров или доступ закрыт.")
        return

    model_id = MODEL_IDS.get(context.user_data.get('model', 'gemini'))
    prompt = f"Сделай краткое содержание текста (источник: {lang}). Выдели 3-5 главных мыслей.\n\nТекст: {text}"
    
    answer = await get_ai_response(model_id, [{"role": "user", "content": prompt}])
    await status_msg.delete()
    await update.message.reply_text(f"📝 *Результат анализа:* \n\n{answer}", parse_mode=ParseMode.MARKDOWN)

# ===== ОБРАБОТЧИКИ =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Бот Meridian готов к работе!", reply_markup=get_main_menu())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id
    
    if text == "🤖 Модели":
        await update.message.reply_text("Выбери модель:", reply_markup=ReplyKeyboardMarkup([list(MODEL_IDS.keys())[:3], list(MODEL_IDS.keys())[3:], ["🔙 Назад"]], resize_keyboard=True))
        return
    if text == "🗑️ Очистить":
        user_chats[user_id] = []
        await update.message.reply_text("🧹 Очищено!")
        return
    if text == "🔙 Назад":
        await start(update, context)
        return
    
    # Смена модели
    if text in MODEL_IDS:
        context.user_data['model'] = text
        await update.message.reply_text(f"✅ Модель: {text}", reply_markup=get_main_menu())
        return

    # Анализ ссылок
    url_match = re.search(r'https?://[^\s]+', text) if text else None
    if url_match:
        await summarize_url(update, context, url_match.group())
        return

    # Обычный чат
    if user_id not in user_chats: user_chats[user_id] = []
    user_chats[user_id].append({"role": "user", "content": text})
    
    model_id = MODEL_IDS.get(context.user_data.get('model', 'gemini'))
    persona = PERSONAS.get(user_persona.get(user_id, "default"))["prompt"]
    
    messages = [{"role": "system", "content": persona}] + user_chats[user_id][-MAX_HISTORY_LENGTH:]
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    answer = await get_ai_response(model_id, messages)
    
    user_chats[user_id].append({"role": "assistant", "content": answer})
    await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)

# ===== ЗАПУСК =====

async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, handle_message))
    
    if URL:
        await app.bot.set_webhook(f"{URL}/telegram")
    
    async def telegram_webhook(request: Request) -> Response:
        json_data = await request.json()
        await app.update_queue.put(Update.de_json(json_data, app.bot))
        return Response()
    
    starlette_app = Starlette(routes=[Route("/telegram", telegram_webhook, methods=["POST"]), Route("/healthcheck", lambda _: PlainTextResponse("ok"), methods=["GET"])])
    
    import uvicorn
    config = uvicorn.Config(app=starlette_app, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    
    async with app:
        await app.start()
        await server.serve()
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
