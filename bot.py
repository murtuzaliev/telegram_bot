import os
import asyncio
import logging
import base64
import urllib.parse
import time
import re
from collections import defaultdict
import httpx
from starlette.applications import Starlette
from starlette.responses import Response, PlainTextResponse
from starlette.requests import Request
from starlette.routing import Route
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from telegram.constants import ParseMode

from openai import OpenAI
from bs4 import BeautifulSoup

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

# Расширенная информация о моделях для меню
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
    "default": {
        "name": "🤖 Ассистент",
        "icon": "🤖",
        "prompt": "Ты полезный и дружелюбный ассистент. Отвечай кратко и по делу."
    },
    "coder": {
        "name": "💻 Python Разработчик",
        "icon": "💻",
        "prompt": "Ты Senior Python разработчик с 10-летним опытом. Отвечай с примерами кода, объясняй решения, используй best practices."
    },
    "translator": {
        "name": "🌍 Профессиональный переводчик",
        "icon": "🌍",
        "prompt": "Ты профессиональный переводчик. Переводи всё на русский язык, сохраняя стиль и смысл. Если нужно, объясняй нюансы перевода."
    },
    "teacher": {
        "name": "📚 Учитель английского",
        "icon": "📚",
        "prompt": "Ты опытный учитель английского языка. Исправляй ошибки ученика, объясняй грамматику, предлагай альтернативные варианты."
    },
    "writer": {
        "name": "✍️ Креативный копирайтер",
        "icon": "✍️",
        "prompt": "Ты креативный копирайтер. Пиши увлекательно, используй метафоры, создавай запоминающиеся тексты."
    },
    "psychologist": {
        "name": "🎭 Коуч-психолог",
        "icon": "🎭",
        "prompt": "Ты поддерживающий психолог и коуч. Задавай уточняющие вопросы, помогай рефлексировать, не давай готовых решений, а веди к ним."
    },
    "smm": {
        "name": "📱 SMM-менеджер",
        "icon": "📱",
        "prompt": "Ты эксперт по SMM. Генерируй идеи для постов, предлагай форматы контента, пиши вовлекающие тексты."
    },
    "analyst": {
        "name": "📊 Бизнес-аналитик",
        "icon": "📊",
        "prompt": "Ты бизнес-аналитик. Структурируй информацию, делай выводы, предлагай метрики для оценки."
    }
}

# Инициализация клиента
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Хранилища
user_chats = {}
user_last_message = defaultdict(float)
user_persona = {}
user_voice_enabled = defaultdict(bool)
MAX_HISTORY_LENGTH = 10

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
        logger.error("Timeout ошибка")
        return "⏰ Превышено время ожидания ответа от AI. Попробуйте позже."
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return f"❌ Ошибка: {str(e)[:100]}"

async def text_to_speech(text, voice="alloy"):
    """Преобразование текста в голосовое сообщение"""
    try:
        text = text[:4096]
        response = await asyncio.to_thread(
            client.audio.speech.create,
            model="tts-1",
            voice=voice,
            input=text
        )
        return response.content
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None

async def voice_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Отправить голосовой ответ"""
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
        audio_bytes = await text_to_speech(text)
        if audio_bytes:
            await update.message.reply_voice(
                voice=audio_bytes,
                caption="🎙️ *Голосовой ответ*",
                parse_mode=ParseMode.MARKDOWN
            )
            return True
    except Exception as e:
        logger.error(f"Voice reply error: {e}")
    return False

async def extract_text_from_url(url: str) -> str:
    """Извлечение текста из веб-страницы"""
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
    """Суммаризация контента по ссылке"""
    status_msg = await update.message.reply_text("🔍 *Анализирую ссылку...*", parse_mode=ParseMode.MARKDOWN)
    
    text = await extract_text_from_url(url)
    if not text:
        await status_msg.edit_text("❌ Не удалось извлечь текст из ссылки. Проверьте, что ссылка открывается.")
        return
    
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key)
    
    prompt = f"""
    Сделай краткую выжимку из следующего текста. Выдели главные мысли, ключевые факты и основные выводы.
    
    Текст:
    {text}
    
    Ответь в формате:
    📌 *Краткое содержание:*
    [3-5 предложений]
    
    🔑 *Ключевые моменты:*
    • [пункт 1]
    • [пункт 2]
    • [пункт 3]
    
    💡 *Выводы:*
    [1-2 предложения]
    """
    
    await status_msg.edit_text("🤔 *Создаю краткое содержание...*", parse_mode=ParseMode.MARKDOWN)
    
    answer = await get_ai_response(model_id, [{"role": "user", "content": prompt}])
    
    if answer:
        await status_msg.delete()
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    else:
        await status_msg.edit_text("❌ Не удалось создать краткое содержание. Попробуйте позже.")

# ===== КОМАНДЫ =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    keyboard = [
        [InlineKeyboardButton("🎨 Создать картинку", callback_data="gen_image_help")],
        [InlineKeyboardButton("🤖 Выбрать модель", callback_data="show_models")],
        [InlineKeyboardButton("🎭 Выбрать личность", callback_data="show_personas")],
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="clear_history")],
        [InlineKeyboardButton("🔊 Голосовые ответы", callback_data="toggle_voice")],
        [InlineKeyboardButton("📊 О боте", callback_data="about_bot")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    current_key = context.user_data.get('model', 'gemini')
    current_name = MODELS.get(current_key, "Gemini")
    
    user_id = update.effective_user.id
    current_persona = user_persona.get(user_id, "default")
    persona_name = PERSONAS[current_persona]["name"]
    voice_status = "включены" if user_voice_enabled[user_id] else "выключены"
    
    text = (
        f"👋 *Привет! Я твой мульти-ассистент.*\n\n"
        f"⚙️ **Текстовая модель:** {current_name}\n"
        f"🎭 **Личность:** {persona_name}\n"
        f"🔊 **Голосовые ответы:** {voice_status}\n\n"
        f"📷 **Я вижу фото** (Gemini/GPT-4o)\n"
        f"🎙️ **Я слышу голос** (Whisper)\n"
        f"🎨 **Я создаю картинки** (бесплатно)\n"
        f"🔗 **Я анализирую ссылки** (отправь ссылку)\n\n"
        "Напиши вопрос, пришли фото/голос/ссылку или нажми кнопку 👇"
    )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📚 *Доступные команды:*

/start - Главное меню
/help - Показать это сообщение
/image [описание] - Создать картинку

*Как пользоваться:*
• Просто отправьте текст - я отвечу
• Отправьте фото с вопросом - я проанализирую
• Отправьте голосовое сообщение - я распознаю и отвечу
• Отправьте ссылку - я сделаю краткое содержание
• Используйте /image для генерации картинок

*Модели:*
🤖 Gemini 2.0 Flash - быстрая, бесплатно
🧠 GPT-4o - поддерживает фото
🎭 Claude 3.5 Sonnet - креативная
💚 GPT-4o Mini - экономичная
🦙 Llama 3.3 - бесплатная
🐋 DeepSeek V3 - мощная, бесплатно

*Личности:*
Можно выбрать разные стили общения: программист, переводчик, учитель, копирайтер, психолог, SMM-менеджер, аналитик
    """
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /image [промпт]"""
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ *Ошибка:* напишите описание после команды.\n\n"
            "Пример:\n`/image красный соус, горы, арт`\n"
            "`/image логотип Meridian, минимализм`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    prompt = " ".join(context.args)
    
    status_msg = await update.message.reply_text(
        f"⏳ *Генерирую изображение:* `{prompt}`...",
        parse_mode=ParseMode.MARKDOWN
    )
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&private=true&width=1024&height=1024"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client_http:
            response = await client_http.get(image_url)
            
            if response.status_code == 200:
                await update.message.reply_photo(
                    photo=response.content,
                    caption=f"🎨 *Результат для:* `{prompt}`",
                    parse_mode=ParseMode.MARKDOWN
                )
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            else:
                await status_msg.edit_text("❌ Бесплатный сервис генерации временно недоступен. Попробуйте позже.")
    except httpx.TimeoutException:
        await status_msg.edit_text("⏰ Превышено время ожидания. Попробуйте позже.")
    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        await status_msg.edit_text(f"❌ Ошибка при генерации: {str(e)[:100]}")

# ===== ОБРАБОТЧИКИ СООБЩЕНИЙ =====

async def handle_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текста, фото, голоса и ссылок"""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Антиспам
    current_time = time.time()
    if current_time - user_last_message[user_id] < 1:
        await update.message.reply_text("⏳ Пожалуйста, не спамьте! Подождите секунду.")
        return
    user_last_message[user_id] = current_time
    
    # Проверяем, является ли сообщение ссылкой
    if update.message.text and re.match(r'https?://[^\s]+', update.message.text):
        url_match = re.search(r'https?://[^\s]+', update.message.text)
        if url_match:
            await summarize_url(update, context, url_match.group())
            return
    
    # Инициализация истории
    if user_id not in user_chats:
        user_chats[user_id] = []
    
    # Получаем текущую модель и личность
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key, "google/gemini-2.0-flash-001")
    persona_key = user_persona.get(user_id, "default")
    persona_prompt = PERSONAS[persona_key]["prompt"]
    
    content = []
    user_text = ""
    
    # Обработка разных типов сообщений
    if update.message.text:
        user_text = update.message.text
        content = [{"type": "text", "text": user_text}]
        
    elif update.message.photo:
        if model_id not in VISION_MODELS:
            await update.message.reply_text(
                f"❌ Модель {MODELS.get(model_key, model_key)} не умеет анализировать фото.\n"
                "Переключитесь на Gemini или GPT-4o с помощью кнопки 'Выбрать модель'."
            )
            return
        
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode('utf-8')
        caption = update.message.caption or "Что на этой картинке?"
        user_text = f"[Фото] {caption}"
        content = [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
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
                    file=audio_file
                )
            user_text = transcript.text
            await update.message.reply_text(
                f"🎤 *Распознано:* {user_text}",
                parse_mode=ParseMode.MARKDOWN
            )
            content = [{"type": "text", "text": user_text}]
        except Exception as e:
            logger.error(f"Whisper Error: {e}")
            await update.message.reply_text("❌ Ошибка распознавания голоса. Проверьте ключ API.")
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if not content:
        return
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    # Сохраняем сообщение пользователя
    user_chats[user_id].append({"role": "user", "content": content})
    
    # Подготавливаем историю для API с системным промптом
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
    
    # Сохраняем ответ
    user_chats[user_id].append({"role": "assistant", "content": answer})
    
    # Отправляем текстовый ответ
    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i+4096])
    else:
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    
    # Отправляем голосовой ответ, если включено
    if user_voice_enabled[user_id]:
        await voice_reply(update, context, answer)

# ===== МЕНЮ И КНОПКИ =====

async def show_models_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Показать меню выбора моделей с иконками и описаниями"""
    keyboard = []
    keys = list(MODELS_INFO.keys())
    
    for i in range(0, len(keys), 2):
        row = []
        for j in range(2):
            if i + j < len(keys):
                key = keys[i + j]
                info = MODELS_INFO[key]
                vision_badge = " 📷" if info["vision"] else ""
                button_text = f"{info['icon']} {info['name']}{vision_badge}"
                row.append(InlineKeyboardButton(button_text, callback_data=f"model_{key}"))
        if row:
            keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("ℹ️ О моделях", callback_data="models_info")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
    
    text = (
        "🎯 *Выберите текстовую модель*\n\n"
        "📷 *Vision* — видят фото\n"
        "💎 *Бесплатные*: Gemini, Llama, DeepSeek\n"
        "✨ *GPT-4o и Claude* — премиум"
    )
    
    if query:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def models_info(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Информация о моделях"""
    info_text = """
📊 *Сравнение моделей:*

🤖 *Gemini 2.0 Flash*
• Бесплатно
• Быстрые ответы
• Поддерживает фото

🧠 *GPT-4o*
• Платная (но мощная)
• Лучшее понимание фото
• Отличный код

🎭 *Claude 3.5 Sonnet*
• Платная
• Креативные ответы
• Длинный контекст

💚 *GPT-4o Mini*
• Дешевая
• Быстрая
• Хороша для простых задач

🦙 *Llama 3.3*
• Полностью бесплатно
• Хороший баланс

🐋 *DeepSeek V3*
• Полностью бесплатно
• Мощная, как GPT-4

💡 *Совет:* Для фото используйте Gemini или GPT-4o
    """
    if query:
        await query.message.reply_text(info_text, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(info_text, parse_mode=ParseMode.MARKDOWN)

async def show_personas_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Показать меню выбора личности"""
    keyboard = []
    keys = list(PERSONAS.keys())
    
    for i in range(0, len(keys), 2):
        row = []
        for j in range(2):
            if i + j < len(keys):
                key = keys[i + j]
                info = PERSONAS[key]
                row.append(InlineKeyboardButton(
                    f"{info['icon']} {info['name']}",
                    callback_data=f"persona_{key}"
                ))
        if row:
            keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
    
    user_id = update.effective_user.id if update.effective_user else query.from_user.id
    current_persona = user_persona.get(user_id, "default")
    current_name = PERSONAS[current_persona]["name"]
    
    text = (
        f"🎭 *Выберите личность ИИ*\n\n"
        f"Текущая: {current_name}\n\n"
        f"Каждая личность меняет стиль общения и подход к ответам."
    )
    
    if query:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def toggle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Включить/выключить голосовые ответы"""
    user_id = update.effective_user.id if update.effective_user else query.from_user.id
    user_voice_enabled[user_id] = not user_voice_enabled[user_id]
    status = "включены" if user_voice_enabled[user_id] else "выключены"
    
    text = f"🔊 *Голосовые ответы {status}!*"
    if query:
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def about_bot(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Информация о боте"""
    about_text = """
🤖 *О боте*

Версия: 2.0
Разработчик: @your_username

*Возможности:*
• 6 моделей ИИ (Gemini, GPT-4o, Claude и др.)
• 8 личностей (программист, переводчик, учитель и др.)
• Распознавание и генерация голоса
• Анализ фото (Vision)
• Генерация изображений
• Анализ ссылок (суммаризация)
• История диалога

*Технологии:*
• Python + python-telegram-bot
• OpenRouter API
• OpenAI Whisper & TTS
• Pollinations.ai

*Контакты:*
По вопросам и предложениям пишите @your_username
    """
    if query:
        await query.message.reply_text(about_text, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(about_text, parse_mode=ParseMode.MARKDOWN)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки"""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    if query.data == "show_models":
        await show_models_menu(update, context, query)
        
    elif query.data == "models_info":
        await models_info(update, context, query)
        
    elif query.data == "show_personas":
        await show_personas_menu(update, context, query)
        
    elif query.data == "gen_image_help":
        await query.message.reply_text(
            "🎨 *Как создать картинку бесплатно:*\n\n"
            "Используйте команду `/image` и напишите описание.\n\n"
            "Примеры:\n"
            "`/image логотип Meridian соус, минимализм, вектор`\n"
            "`/image красный закат над морем, цифровой арт`\n\n"
            "Картинки генерируются автоматически!",
            parse_mode=ParseMode.MARKDOWN
        )
        
    elif query.data == "clear_history":
        user_chats[user_id] = []
        await query.edit_message_text("🧹 *История диалога очищена!*", parse_mode=ParseMode.MARKDOWN)
        
    elif query.data == "toggle_voice":
        await toggle_voice(update, context, query)
        
    elif query.data == "about_bot":
        await about_bot(update, context, query)
        
    elif query.data.startswith("model_"):
        new_model = query.data.replace("model_", "")
        context.user_data['model'] = new_model
        text = f"✅ *Установлена модель:* **{MODELS_INFO[new_model]['name']}**\n\n{ MODELS_INFO[new_model]['desc'] }"
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
        
    elif query.data.startswith("persona_"):
        persona_key = query.data.replace("persona_", "")
        user_persona[user_id] = persona_key
        persona_name = PERSONAS[persona_key]["name"]
        await query.edit_message_text(
            f"✅ *Установлена личность:* {persona_name}\n\n"
            f"Теперь я буду общаться в этом стиле!",
            parse_mode=ParseMode.MARKDOWN
        )
        
    elif query.data == "back_to_menu":
        await start(update, context)

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
    app.add_handler(CallbackQueryHandler(button_callback))
    
    # Обрабатываем текст, фото и голос
    app.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.VOICE,
        handle_all_messages
    ))
    
    # Установка webhook
    if URL:
        webhook_url = f"{URL}/telegram"
        await app.bot.set_webhook(webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")
    else:
        logger.warning("RENDER_EXTERNAL_URL не установлен, webhook не будет работать!")
    
    # Создаем Starlette приложение
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
