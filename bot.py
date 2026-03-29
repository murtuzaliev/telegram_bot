import os
import asyncio
import logging
from starlette.applications import Starlette
from starlette.responses import Response, PlainTextResponse
from starlette.requests import Request
from starlette.routing import Route
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from openai import OpenAI

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ПЕРЕМЕННЫЕ =====
TOKEN = os.environ["TELEGRAM_TOKEN"]           # Токен Telegram
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]  # Ключ OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
URL = os.environ["RENDER_EXTERNAL_URL"]        # Render даст сам
PORT = int(os.getenv("PORT", 8000))
# =======================

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
user_chats = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.message.text
    
    try:
        history = user_chats.get(user_id, [])
        history.append({"role": "user", "content": msg})
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="google/gemini-2.5-flash",  # рабочая модель
            messages=history,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        user_chats[user_id] = history[-10:]
        
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"😕 Ошибка: {str(e)[:300]}")

async def main():
    # Создаём приложение Telegram
    app = Application.builder().token(TOKEN).updater(None).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Устанавливаем вебхук (вместо polling)
    await app.bot.set_webhook(f"{URL}/telegram", allowed_updates=Update.ALL_TYPES)
    logger.info(f"Webhook set to {URL}/telegram")
    
    # Создаём Starlette сервер
    async def telegram_webhook(request: Request) -> Response:
        await app.update_queue.put(Update.de_json(await request.json(), app.bot))
        return Response()
    
    async def health_check(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")
    
    starlette = Starlette(routes=[
        Route("/telegram", telegram_webhook, methods=["POST"]),
        Route("/healthcheck", health_check, methods=["GET"]),
    ])
    
    # Запускаем сервер
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app=starlette, host="0.0.0.0", port=PORT))
    async with app:
        await app.start()
        await server.serve()
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())