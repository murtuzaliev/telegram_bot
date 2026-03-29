import os
import asyncio
import logging
import base64
from starlette.applications import Starlette
from starlette.responses import Response, PlainTextResponse
from starlette.requests import Request
from starlette.routing import Route
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler

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

# ===== НАСТРОЙКИ МОДЕЛЕЙ =====
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
    keyboard = [
        [InlineKeyboardButton("🤖 Выбрать модель", callback_data="show_models")],
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="clear_history")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    current_key = context.user_data.get('model', 'gemini')
    current_name = MODELS.get(current_key, "Gemini")
    
    await update.message.reply_text(
        f"👋 *Бот обновлен!*\n\n"
        f"⚙️ **Модель:** {current_name}\n"
        f"📷 **Я вижу фото** и перевожу текст с них.\n"
        f"🎙️ **Я слышу голос** и отвечаю на аудио.\n\n"
        "Пришли фото, голос или текст 👇",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

# ===== ОБРАБОТЧИКИ СООБЩЕНИЙ =====

async def handle_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # 1. Инициализация истории
    if user_id not in user_chats:
        user_chats[user_id] = []
    
    model_key = context.user_data.get('model', 'gemini')
    model_id = MODEL_IDS.get(model_key, "google/gemini-2.0-flash-001")
    
    content = []
    user_text = ""

    # 2. Обработка ТЕКСТА
    if update.message.text:
        user_text = update.message.text
        content.append({"type": "text", "text": user_text})

    # 3. Обработка ФОТО
    elif update.message.photo:
        if model_id not in VISION_MODELS:
            await update.message.reply_text(f"❌ Модель {model_key} не умеет анализировать фото. Переключитесь на Gemini или GPT-4o.")
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode('utf-8')
        
        caption = update.message.caption or "Что на этой картинке? Если есть английский текст - переведи."
        user_text = f"[Фото] {caption}"
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    # 4. Обработка ГОЛОСА
    elif update.message.voice:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        voice_file = await update.message.voice.get_file()
        voice_bytes = await voice_file.download_as_bytearray()
        
        # Сохраняем временно для отправки в Whisper
        with open("voice.ogg", "wb") as f:
            f.write(voice_bytes)
        
        try:
            # Используем стандартный эндпоинт OpenAI для Whisper через OpenRouter
            with open("voice.ogg", "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="openai/whisper-1", 
                    file=audio_file
                )
            user_text = transcript.text
            await update.message.reply_text(f"🎤 _Распознано:_ {user_text}", parse_mode="Markdown")
            content.append({"type": "text", "text": user_text})
        except Exception as e:
            logger.error(f"Whisper Error: {e}")
            await update.message.reply_text("❌ Не удалось распознать голос.")
            return

    # 5. Отправка в AI
    if not content: return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    # Добавляем в историю
    user_chats[user_id].append({"role": "user", "content": content})
    
    # Ограничиваем историю (только текст для экономии токенов в истории)
    messages_for_api = []
    for m in user_chats[user_id][-MAX_HISTORY_LENGTH:]:
        # Для истории оставляем только текст, чтобы не пересылать картинки по 10 раз
        if isinstance(m["content"], list):
            text_val = next((item["text"] for item in m["content"] if item["type"] == "text"), "")
            messages_for_api.append({"role": m["role"], "content": text_val})
        else:
            messages_for_api.append(m)

    # Если это текущее сообщение с картинкой — заменяем текст на полный контент (текст + фото)
    messages_for_api[-1]["content"] = content

    answer = await get_ai_response(model_id, messages_for_api)
    
    if not answer:
        answer = "⚠️ Ошибка связи с AI. Попробуйте еще раз."

    # Сохраняем ответ в историю
    user_chats[user_id].append({"role": "assistant", "content": answer})
    
    # Отправка пользователю
    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i+4096])
    else:
        await update.message.reply_text(answer, parse_mode="Markdown")

# ===== КНОПКИ И МЕНЮ =====

async def show_models_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    keyboard = []
    keys = list(MODELS.keys())
    for i in range(0, len(keys), 2):
        row = [InlineKeyboardButton(MODELS[keys[i+j]], callback_data=f"model_{keys[i+j]}") for j in range(2) if i+j < len(keys)]
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
    
    text = "🎯 **Выберите модель:**\n_(Gemini и GPT-4o поддерживают фото)_"
    if query:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    if query.data == "show_models":
        await show_models_menu(update, context, query)
    elif query.data == "clear_history":
        user_chats[user_id] = []
        await query.answer("🧹 История очищена")
    elif query.data.startswith("model_"):
        new_model = query.data.replace("model_", "")
        context.user_data['model'] = new_model
        await query.edit_message_text(f"✅ Установлена модель: **{MODELS[new_model]}**", parse_mode="Markdown")
    elif query.data == "back_to_menu":
        await start(update, context)

# ===== СЕРВЕРНАЯ ЧАСТЬ =====

async def main():
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_callback))
    # Обрабатываем текст, фото и голос
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.VOICE, handle_all_messages))
    
    await app.bot.set_webhook(f"{URL}/telegram")
    
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
