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
from youtube_transcript_api.formatters import TextFormatter

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ПЕРЕМЕННЫЕ =====
TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.getenv("PORT", 8000))

# Проверка наличия обязательных переменных
if not TOKEN:
    logger.error("TELEGRAM_TOKEN не установлен!")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY не установлен!")

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

# Расширенная информация о моделях
MODELS_INFO = {
    "gemini": {"icon": "🤖", "name": "Gemini 2.0 Flash", "desc": "Быстрая, умная, бесплатно", "vision": True},
    "gpt-4o": {"icon": "🧠", "name": "GPT-4o", "desc": "Видит фото, мощная", "vision": True},
    "claude": {"icon": "🎭", "name": "Claude 3.5 Sonnet", "desc": "Креативная, длинные ответы", "vision": True},
    "gpt-mini": {"icon": "💚", "name": "GPT-4o Mini", "desc": "Экономичная, быстрая", "vision": True},
    "llama": {"icon": "🦙", "name": "Llama 3.3", "desc": "Совсем бесплатно", "vision": False},
    "deepseek": {"icon": "🐋", "name": "DeepSeek V3", "desc": "Мощная, бесплатно", "vision": False},
}

# Список моделей, которые умеют работать с картинками
VISION_MODELS = ["google/gemini-2.0-flash-001", "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]

# ===== СИСТЕМА ЛИЧНОСТЕЙ =====
PERSONAS = {
    "default": {"name": "🤖 Ассистент", "icon": "🤖", "prompt": "Ты полезный и дружелюбный ассистент. Отвечай кратко и по делу."},
    "coder": {"name": "💻 Python Dev", "icon": "💻", "prompt": "Ты Senior Python разработчик. Отвечай с примерами кода, объясняй решения."},
    "translator": {"name": "🌍 Переводчик", "icon": "🌍", "prompt": "Ты профессиональный переводчик. Переводи всё на русский язык."},
    "teacher": {"name": "📚 Учитель", "icon": "📚", "prompt": "Ты учитель английского. Исправляй ошибки, объясняй грамматику."},
    "writer": {"name": "✍️ Копирайтер", "icon": "✍️", "prompt": "Ты креативный копирайтер. Пиши увлекательно, используй метафоры."},
    "psychologist": {"name": "🎭 Психолог", "icon": "🎭", "prompt": "Ты поддерживающий психолог. Задавай уточняющие вопросы, помогай рефлексировать."},
    "smm": {"name": "📱 SMM", "icon": "📱", "prompt": "Ты эксперт по SMM. Генерируй идеи для постов, пиши вовлекающие тексты."},
    "analyst": {"name": "📊 Аналитик", "icon": "📊", "prompt": "Ты бизнес-аналитик. Структурируй информацию, делай выводы."}
}

# Групповой режим: настройки
GROUP_CONFIG = {
    "enabled": True,
    "spontaneous_chance": 12,
    "max_context_messages": 8,
    "min_message_length": 30,
}

# Хранилища
user_chats = {}
user_last_message = defaultdict(float)
user_persona = {}
group_messages = defaultdict(list)
MAX_HISTORY_LENGTH = 10

# Инициализация клиента
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ===== МЕНЮ (REPLY KEYBOARD) =====

def get_main_menu() -> ReplyKeyboardMarkup:
    """Главное меню (всегда внизу)"""
    keyboard = [
        ["🤖 Модели", "🎭 Личности"],
        ["🎨 Картинка", "🔗 Анализ ссылки"],
        ["🗑️ Очистить", "ℹ️ Помощь"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

def get_models_menu() -> ReplyKeyboardMarkup:
    """Меню выбора моделей"""
    keyboard = [
        ["🤖 Gemini", "🧠 GPT-4o"],
        ["🎭 Claude", "💚 GPT-4o Mini"],
        ["🦙 Llama", "🐋 DeepSeek"],
        ["🔙 Назад"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

def get_personas_menu() -> ReplyKeyboardMarkup:
    """Меню выбора личностей"""
    keyboard = [
        ["🤖 Ассистент", "💻 Python Dev"],
        ["🌍 Переводчик", "📚 Учитель"],
        ["✍️ Копирайтер", "🎭 Психолог"],
        ["📱 SMM", "📊 Аналитик"],
        ["🔙 Назад"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

async def get_ai_response(model_id, messages):
    """Универсальная функция запроса к AI"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id,
            messages=messages,
            max_tokens=1500,
            timeout=30.0
        )
        return response.choices[0].message.content
    except httpx.TimeoutException:
        return "⏰ Превышено время ожидания. Попробуйте позже."
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return f"❌ Ошибка: {str(e)[:100]}"

def extract_youtube_transcript(url: str) -> tuple:
    """Извлечение субтитров из YouTube видео. Возвращает (текст, язык)"""
    try:
        # Извлекаем ID видео из URL
        video_id = None
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "shorts/" in url:
                video_id = url.split("shorts/")[1].split("?")[0]
        
        if not video_id:
            return None, None
        
        # Получаем список доступных транскрипций
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Пробуем получить русские субтитры
        try:
            transcript = transcript_list.find_transcript(['ru', 'uk'])
            language = "русский"
        except:
            try:
                transcript = transcript_list.find_transcript(['en'])
                language = "английский"
            except:
                # Если нет, берем любые доступные
                available = transcript_list._transcripts
                if available:
                    transcript = list(available.values())[0]
                    language = transcript.language
                else:
                    return None, None
        
        # Форматируем в текст
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript.fetch())
        
        # Ограничиваем длину
        return text[:4000], language
        
    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return None, None

async def extract_text_from_url(url: str) -> str:
    """Извлечение текста из обычной веб-страницы"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client_http:
            response = await client_http.get(url, follow_redirects=True)
            response.raise_for_status()
            
            if 'charset' in response.headers.get('content-type', ''):
                encoding = response.headers['content-type'].split('charset=')[-1]
            else:
                encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=encoding)
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:4000]
    except Exception as e:
        logger.error(f"URL extraction error: {e}")
        return None

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    """Суммаризация контента по ссылке (поддержка YouTube)"""
    status_msg = await update.message.reply_text("🔍 *Анализирую ссылку...*", parse_mode=ParseMode.MARKDOWN)
    
    text = None
    is_youtube = "youtube.com" in url or "youtu.be" in url
    language = None
    
    if is_youtube:
        await status_msg.edit_text("🎬 *Обнаружено YouTube видео!*\nИзвлекаю субтитры...", parse_mode=ParseMode.MARKDOWN)
        text, language = extract_youtube_transcript(url)
        source = "YouTube видео"
    else:
        text = await extract_text_from_url(url)
        source = "веб-страницы"
    
    if not text:
        if is_youtube:
            await status_msg.edit_text(
                "❌ *Не удалось извлечь субтитры из YouTube видео*\n\n"
                "Возможные причины:\n"
                "• У видео нет субтитров\n"
                "• Видео на недоступном языке\n"
                "• Это короткое видео (Shorts) без субтитров\n\n"
                "Попробуйте другое видео с субтитрами.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await status_msg.edit_text(
                "❌ *Не удалось извлечь текст из ссылки*\n\n"
                "Проверьте, что ссылка открывается и содержит текст.",
                parse_mode=ParseMode.MARKDOWN
            )
        return
    
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key)
    
    # Разный промпт для YouTube и обычных ссылок
    if is_youtube:
        prompt = f"""
        Это транскрипция (субтитры) YouTube видео на {language} языке.
        
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

# ===== ОБРАБОТЧИК МЕНЮ =====

async def handle_menu_commands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обработка команд из постоянного меню"""
    text = update.message.text
    user_id = update.effective_user.id
    
    # Главное меню
    if text == "🤖 Модели":
        await update.message.reply_text(
            "🎯 *Выберите модель:*\n\n"
            "📷 *Vision* — видят фото\n"
            "💎 *Бесплатные*: Gemini, Llama, DeepSeek",
            reply_markup=get_models_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
        
    elif text == "🎭 Личности":
        await update.message.reply_text(
            "🎭 *Выберите личность:*\n\n"
            "Каждая личность меняет стиль общения и подход к ответам.",
            reply_markup=get_personas_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
        
    elif text == "🎨 Картинка":
        await update.message.reply_text(
            "🎨 *Создание картинки*\n\n"
            "Просто напишите: `/image описание`\n\n"
            "Примеры:\n"
            "`/image кот в космосе`\n"
            "`/image логотип минимализм`\n"
            "`/image закат над морем, цифровой арт`",
            parse_mode=ParseMode.MARKDOWN
        )
        return True
        
    elif text == "🔗 Анализ ссылки":
        await update.message.reply_text(
            "🔗 *Анализ ссылки*\n\n"
            "Просто отправьте любую ссылку, и я сделаю краткое содержание!\n\n"
            "Поддерживается:\n"
            "• YouTube видео (с субтитрами)\n"
            "• Веб-страницы\n"
            "• Статьи\n"
            "• Новости"
        )
        return True
        
    elif text == "🗑️ Очистить":
        user_chats[user_id] = []
        await update.message.reply_text(
            "🧹 *История диалога очищена!*",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
        
    elif text == "ℹ️ Помощь":
        await help_command(update, context)
        return True
        
    elif text == "🔙 Назад":
        await update.message.reply_text(
            "👋 *Главное меню*",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
    
    # Выбор моделей
    model_map = {
        "🤖 Gemini": "gemini",
        "🧠 GPT-4o": "gpt-4o",
        "🎭 Claude": "claude",
        "💚 GPT-4o Mini": "gpt-mini",
        "🦙 Llama": "llama",
        "🐋 DeepSeek": "deepseek"
    }
    if text in model_map:
        model_key = model_map[text]
        context.user_data['model'] = model_key
        model_name = MODELS_INFO[model_key]["name"]
        await update.message.reply_text(
            f"✅ *Установлена модель:* {model_name}\n\n{MODELS_INFO[model_key]['desc']}",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
    
    # Выбор личностей
    persona_map = {
        "🤖 Ассистент": "default",
        "💻 Python Dev": "coder",
        "🌍 Переводчик": "translator",
        "📚 Учитель": "teacher",
        "✍️ Копирайтер": "writer",
        "🎭 Психолог": "psychologist",
        "📱 SMM": "smm",
        "📊 Аналитик": "analyst"
    }
    if text in persona_map:
        persona_key = persona_map[text]
        user_persona[user_id] = persona_key
        persona_name = PERSONAS[persona_key]["name"]
        await update.message.reply_text(
            f"✅ *Установлена личность:* {persona_name}",
            reply_markup=get_main_menu(),
            parse_mode=ParseMode.MARKDOWN
        )
        return True
    
    return False

# ===== КОМАНДЫ =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    
    # Получаем текущие настройки
    model_key = context.user_data.get('model', 'gemini')
    current_model = MODELS_INFO[model_key]["name"]
    current_persona = user_persona.get(user_id, "default")
    persona_name = PERSONAS[current_persona]["name"]
    
    await update.message.reply_text(
        f"👋 *Привет! Я твой AI-ассистент!*\n\n"
        f"⚙️ **Модель:** {current_model}\n"
        f"🎭 **Личность:** {persona_name}\n\n"
        f"Я умею:\n"
        f"📝 Отвечать на вопросы\n"
        f"🎨 Генерировать картинки\n"
        f"🔗 Анализировать ссылки (включая YouTube!)\n"
        f"📷 Распознавать фото\n"
        f"🎙️ Слышать голос\n\n"
        f"👇 *Используй меню внизу для навигации!*",
        reply_markup=get_main_menu(),
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
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
• **Отправьте ссылку на YouTube** - я сделаю краткое содержание видео! 🎬
• Отправьте любую другую ссылку - я сделаю выжимку

*В группах:*
• Упомяните меня @{context.bot.username} - я отвечу
• Иногда я сам вступаю в разговор 🎭

*YouTube возможности:*
• Извлекаю субтитры из видео
• Делаю краткое содержание
• Выделяю ключевые моменты
• Работает с русскими и английскими субтитрами
    """
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /image"""
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ *Ошибка:* напишите описание после команды.\n\n"
            "Пример:\n`/image красный закат над морем`\n`/image логотип минимализм`\n`/image кот в космосе`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    prompt = " ".join(context.args)
    
    status_msg = await update.message.reply_text(
        f"⏳ *Генерирую:* `{prompt}`...",
        parse_mode=ParseMode.MARKDOWN
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
                    parse_mode=ParseMode.MARKDOWN
                )
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            else:
                await status_msg.edit_text("❌ Сервис генерации временно недоступен.")
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)[:100]}")

# ===== ОБРАБОТЧИК ЛИЧНЫХ СООБЩЕНИЙ =====

async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка личных сообщений"""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Сначала проверяем команды меню
    if await handle_menu_commands(update, context):
        return
    
    # Антиспам
    current_time = time.time()
    if current_time - user_last_message[user_id] < 1:
        await update.message.reply_text("⏳ Подождите секунду...")
        return
    user_last_message[user_id] = current_time
    
    # Проверяем ссылку (включая YouTube)
    if update.message.text and re.match(r'https?://[^\s]+', update.message.text):
        url_match = re.search(r'https?://[^\s]+', update.message.text)
        if url_match:
            await summarize_url(update, context, url_match.group())
            return
    
    # Инициализация истории
    if user_id not in user_chats:
        user_chats[user_id] = []
    
    # Получаем модель и личность
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key)
    persona_key = user_persona.get(user_id, "default")
    persona_prompt = PERSONAS[persona_key]["prompt"]
    
    content = []
    
    # Обработка текста
    if update.message.text:
        content = [{"type": "text", "text": update.message.text}]
        
    # Обработка фото
    elif update.message.photo:
        if model_id not in VISION_MODELS:
            await update.message.reply_text(
                f"❌ Модель {MODELS_INFO[model_key]['name']} не умеет анализировать фото.\n"
                "Выберите Gemini или GPT-4o в меню 'Модели'."
            )
            return
        
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode('utf-8')
        caption = update.message.caption or "Что на этой картинке?"
        content = [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        
    # Обработка голоса
    elif update.message.voice:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        voice_file = await update.message.voice.get_file()
        file_path = f"voice_{user_id}_{int(time.time())}.ogg"
        
        try:
            await voice_file.download_to_drive(file_path)
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            await update.message.reply_text(
                f"🎤 *Распознано:* {transcript.text}",
                parse_mode=ParseMode.MARKDOWN
            )
            content = [{"type": "text", "text": transcript.text}]
        except Exception as e:
            logger.error(f"Whisper Error: {e}")
            await update.message.reply_text("❌ Ошибка распознавания голоса.")
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if not content:
        return
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    # Сохраняем историю
    user_chats[user_id].append({"role": "user", "content": content})
    
    # Подготавливаем запрос
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
    
    # Получаем ответ
    answer = await get_ai_response(model_id, messages_for_api)
    if not answer:
        answer = "⚠️ Ошибка связи с AI. Попробуйте позже."
    
    user_chats[user_id].append({"role": "assistant", "content": answer})
    
    # Отправляем ответ
    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i+4096])
    else:
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)

# ===== ОБРАБОТЧИК ГРУППОВЫХ СООБЩЕНИЙ =====

async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка сообщений в группах"""
    message = update.message
    chat_id = message.chat_id
    
    if message.from_user and message.from_user.is_bot:
        return
    
    # Сохраняем сообщение
    user_name = message.from_user.first_name if message.from_user else "Пользователь"
    text = message.text or message.caption or ""
    
    group_messages[chat_id].append({
        "name": user_name,
        "text": text,
        "timestamp": time.time()
    })
    
    if len(group_messages[chat_id]) > 50:
        group_messages[chat_id] = group_messages[chat_id][-50:]
    
    # Проверяем, нужно ли отвечать
    bot_username = (await context.bot.get_me()).username
    should_reply = False
    reply_type = "none"
    
    if message.text and f"@{bot_username}" in message.text:
        should_reply = True
        reply_type = "mention"
    elif message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.is_bot:
        if message.reply_to_message.from_user.username == bot_username:
            should_reply = True
            reply_type = "reply"
    elif len(text) >= GROUP_CONFIG["min_message_length"] and random.randint(1, 100) <= GROUP_CONFIG["spontaneous_chance"]:
        should_reply = True
        reply_type = "spontaneous"
    
    if not should_reply:
        return
    
    # Собираем контекст
    context_lines = []
    for msg in group_messages[chat_id][-GROUP_CONFIG["max_context_messages"]:]:
        if msg["text"]:
            context_lines.append(f"{msg['name']}: {msg['text']}")
    
    if not context_lines:
        return
    
    context_text = "\n".join(context_lines)
    
    # Генерируем ответ
    prompts = {
        "mention": f"Ты участник чата. Тебя упомянули. Ответь весело, коротко (1-2 предложения). Контекст:\n{context_text}",
        "reply": f"Ты отвечаешь на сообщение. Будь остроумным. Контекст:\n{context_text}",
        "spontaneous": f"Ты активный участник чата. Напиши короткую шутку или комментарий (1-2 предложения). Контекст:\n{context_text}"
    }
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    answer = await get_ai_response(
        MODEL_IDS["llama"],
        [{"role": "user", "content": prompts.get(reply_type, prompts["spontaneous"])}]
    )
    
    if not answer:
        jokes = ["😏 Интересно...", "🤔 Продолжайте!", "🍿 Наблюдаю..."]
        answer = random.choice(jokes)
    
    await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)

# ===== СЕРВЕРНАЯ ЧАСТЬ =====

async def main():
    """Главная функция запуска бота"""
    if not TOKEN or not OPENROUTER_API_KEY:
        logger.error("Не хватает переменных окружения!")
        return
    
    app = Application.builder().token(TOKEN).build()
    
    # Регистрация команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("image", generate_image_command))
    
    # Обработчики сообщений
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        handle_private_message
    ))
    app.add_handler(MessageHandler(
        filters.PHOTO & filters.ChatType.PRIVATE,
        handle_private_message
    ))
    app.add_handler(MessageHandler(
        filters.VOICE & filters.ChatType.PRIVATE,
        handle_private_message
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP),
        handle_group_message
    ))
    
    # Установка webhook
    if URL:
        webhook_url = f"{URL}/telegram"
        await app.bot.set_webhook(webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")
    
    # Starlette приложение
    async def telegram_webhook(request: Request) -> Response:
        json_data = await request.json()
        await app.update_queue.put(Update.de_json(json_data, app.bot))
        return Response()
    
    starlette_app = Starlette(routes=[
        Route("/telegram", telegram_webhook, methods=["POST"]),
        Route("/healthcheck", lambda _: PlainTextResponse("ok"), methods=["GET"]),
    ])
    
    import uvicorn
    config = uvicorn.Config(app=starlette_app, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    
    async with app:
        await app.start()
        await server.serve()
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
