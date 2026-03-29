import os
import asyncio
import logging
import base64
import httpx  # Библиотека для запросов к бесплатному API картинок
from starlette.applications import Starlette
from starlette.responses import Response, PlainTextResponse
from starlette.requests import Request
from starlette.routing import Route
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from telegram.constants import ParseMode

from openai import OpenAI

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ПЕРЕМЕННЫЕ =====
TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.getenv("PORT", 8000))

# ===== НАСТРОЙКИ МОДЕЛЕЙ (Текст) =====
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

# Список моделей, которые ТОЧНО умеют работать с картинками
VISION_MODELS = ["google/gemini-2.0-flash-001", "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

user_chats = {}
MAX_HISTORY_LENGTH = 10 

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

async def get_ai_response(model_id, messages):
    """Универсальная функция запроса к AI"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id,
            messages=messages,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return None

# ===== КОМАНДЫ =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Улучшенное главное меню
    keyboard = [
        [InlineKeyboardButton("🖼️ Создать картинку (Бесплатно)", callback_data="gen_image_help")],
        [InlineKeyboardButton("🤖 Выбрать модель", callback_data="show_models")],
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="clear_history")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    current_key = context.user_data.get('model', 'gemini')
    current_name = MODELS.get(current_key, "Gemini")
    
    text = (
        f"👋 *Привет! Я твой мульти-ассистент.*\n\n"
        f"⚙️ **Текстовая модель:** {current_name}\n"
        f"📷 **Я вижу фото** (Gemini/GPT-4o).\n"
        f"🎙️ **Я слышу голос** (Whisper).\n"
        f"🎨 **Я создаю картинки** (бесплатно).\n\n"
        "Напиши вопрос, пришли фото/голос или нажми кнопку 👇"
    )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /image [промпт]"""
    chat_id = update.effective_chat.id
    
    # 1. Проверяем, есть ли описание
    if not context.args:
        await update.message.reply_text("❌ Ошибка: напишите описание после команды.\nПример: `/image красный соус, горы, арт`", parse_mode=ParseMode.MARKDOWN)
        return

    prompt = " ".join(context.args)
    
    # 2. Уведомляем пользователя
    status_msg = await update.message.reply_text(f"⏳ _Генерирую изображение по запросу:_ `{prompt}`...", parse_mode=ParseMode.MARKDOWN)
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    # 3. Запрос к бесплатному API Pollinations.ai
    # Мы кодируем промпт для URL
    encoded_prompt = httpx.URL(prompt).path
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&private=true"

    try:
        async with httpx.AsyncClient() as client_http:
            # Pollinations.ai генерирует картинку прямо при GET-запросе
            response = await client_http.get(image_url, timeout=30.0)
            
            if response.status_code == 200:
                # 4. Отправляем полученную картинку
                # Переводим байты в формат, который понимает Telegram
                await update.message.reply_photo(
                    photo=response.content,
                    caption=f"🎨 **Результат для:** `{prompt}`",
                    parse_mode=ParseMode.MARKDOWN
                )
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            else:
                await status_msg.edit_text("❌ Ошибка: бесплатный сервис генерации сейчас недоступен. Попробуйте позже.")
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        await status_msg.edit_text(f"❌ Произошла ошибка при генерации: {e}")

# ===== ОБРАБОТЧИКИ СООБЩЕНИЙ (Текст, Фото, Голос) =====

async def handle_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # (Этот блок остается без изменений, как в предыдущем ответе)
    # За исключением того, что я добавил InputMediaPhoto для более красивой отправки
    
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if user_id not in user_chats:
        user_chats[user_id] = []
    
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key, "google/gemini-2.0-flash-001")
    
    content = []
    user_text = ""

    if update.message.text:
        user_text = update.message.text
        content.append({"type": "text", "text": user_text})

    elif update.message.photo:
        if model_id not in VISION_MODELS:
            await update.message.reply_text(f"❌ Модель {model_key} не умеет анализировать фото. Переключитесь на Gemini или GPT-4o.")
            return
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes
