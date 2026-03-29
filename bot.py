import os
import asyncio
import logging
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
    "gemini-free": "🤖 Gemini 2.0 Flash",
    "llama": "🦙 Llama 3.3 70B (Free)",
    "mistral": "🌪️ Mistral Small (Free)",
    "gpt-mini": "💚 GPT-4o Mini",
    "claude": "🧠 Claude 3.5 Sonnet",
    "deepseek": "🐋 DeepSeek V3",
}

MODEL_IDS = {
    "gemini-free": "google/gemini-2.0-flash-001",
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "mistral": "mistralai/mistral-small-24b-instruct-2501:free",
    "gpt-mini": "openai/gpt-4o-mini",
    "claude": "anthropic/claude-3.5-sonnet",
    "deepseek": "deepseek/deepseek-chat",
}

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

user_chats = {}
MAX_HISTORY_LENGTH = 10 

# ===== КОМАНДЫ =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🤖 Выбрать модель", callback_data="show_models")],
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="clear_history")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    current_key = context.user_data.get('model', 'gemini-free')
    current_name = MODELS.get(current_key, "Gemini")
    
    await update.message.reply_text(
        f"👋 *Привет! Я твой AI-ассистент.*\n\n"
        f"⚙️ **Модель:** {current_name}\n"
        f"🧠 **Память:** {MAX_HISTORY_LENGTH} сообщений\n\n"
        "Напиши вопрос или выбери модель ниже 👇",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def show_models_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    keyboard = []
    keys = list(MODELS.keys())
    for i in range(0, len(keys), 2):
        row = []
        for j in range(2):
            if i + j < len(keys):
                k = keys[i + j]
                row.append(InlineKeyboardButton(MODELS[k], callback_data=f"model_{k}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "🎯 **Выберите нейросеть:**"
    if query:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# ===== ГЛАВНЫЙ ОБРАБОТЧИК (С РОТАЦИЕЙ ПРИ ОШИБКАХ) =====

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if user_id not in user_chats:
        user_chats[user_id] = []
    history = user_chats[user_id]
    history.append({"role": "user", "content": user_text})
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    model_key = context.user_data.get('model', 'gemini-free')
    model_id = MODEL_IDS.get(model_key, "google/gemini-2.0-flash-001")

    answer = ""
    try:
        # 1. Основная попытка
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_id,
            messages=history,
            max_tokens=1500
        )
        answer = response.choices[0].message.content

    except Exception as e:
        error_str = str(e)
        logger.error(f"Ошибка API ({model_id}): {error_str}")
        
        # 2. Ротация: Если лимит (429) или модель не найдена (404), пробуем запасную
        fallback_models = ["mistralai/mistral-small-24b-instruct-2501:free", "google/gemini-2.0-flash-001"]
        
        for f_id in fallback_models:
            if f_id == model_id: continue
            try:
                logger.info(f"Пробую запасную модель: {f_id}")
                res = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=f_id,
                    messages=history[-3:], # Сокращаем для надежности
                    max_tokens=1000
                )
                answer = f"⚠️ *{model_key} перегружена, ответил через резерв:* \n\n" + res.choices[0].message.content
                break
            except:
                continue
        
        if not answer:
            answer = "❌ Все бесплатные лимиты OpenRouter сейчас исчерпаны. Попробуйте через 5-10 минут."

    # Сохраняем и отправляем
    history.append({"role": "assistant", "content": answer})
    user_chats[user_id] = history
    
    if len(answer) > 4096:
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i+4096])
    else:
        await update.message.reply_text(answer, parse_mode="Markdown")

# ===== КНОПКИ =====

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id
    
    if data == "show_models":
        await show_models_menu(update, context, query)
    elif data == "back_to_menu":
        keyboard = [
            [InlineKeyboardButton("🤖 Выбрать модель", callback_data="show_models")],
            [InlineKeyboardButton("🗑️ Очистить историю", callback_data="clear_history")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
        ]
        await query.edit_message_text("🏠 **Главное меню**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    elif data == "clear_history":
        user_chats[user_id] = []
        await query.answer("✅ История очищена", show_alert=True)
    elif data == "stats":
        h_len = len(user_chats.get(user_id, []))
        await query.answer(f"Сообщений в памяти: {h_len}\nМодель: {context.user_data.get('model', 'gemini-free')}", show_alert=True)
    elif data.startswith("model_"):
        new_model = data.replace("model_", "")
        context.user_data['model'] = new_model
        await query.answer(f"Включено: {MODELS[new_model]}")
        await query.edit_message_text(f"✅ Теперь я использую **{MODELS[new_model]}**", parse_mode="Markdown")

# ===== ЗАПУСК =====

async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    await app.bot.set_webhook(f"{URL}/telegram")
    
    async def telegram_webhook(request: Request) -> Response:
        json_data = await request.json()
        await app.update_queue.put(Update.de_json(json_data, app.bot))
        return Response()
    
    async def health_check(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")
    
    starlette_app = Starlette(routes=[
        Route("/telegram", telegram_webhook, methods=["POST"]),
        Route("/healthcheck", health_check, methods=["GET"]),
    ])
    
    import uvicorn
    config = uvicorn.Config(app=starlette_app, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    
    async with app:
        await app.start()
        await server.serve()
        await app.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass