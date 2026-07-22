"""
Profesyonel Telegram AI Sohbet Botu
Google Gemini API kullanır, hata yönetimi + retry + konuşma geçmişi içerir.
"""

import os
import logging
import asyncio
from collections import defaultdict

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("TELEGRAM_BOT_TOKEN ve GEMINI_API_KEY .env dosyasında tanımlı olmalı.")

MODEL_NAME = "gemini-2.5-flash"
FALLBACK_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_HISTORY_MESSAGES = 20
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("telegram_ai_bot")

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "Sen yardımsever, samimi ve kısa-net cevaplar veren bir Telegram asistanısın. "
    "Türkçe konuşan kullanıcılara Türkçe cevap ver."
)

model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)
fallback_model = genai.GenerativeModel(FALLBACK_MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

user_histories: dict[int, list[dict]] = defaultdict(list)


async def generate_with_retry(chat_history: list[dict], user_message: str) -> str:
    for use_model, label in ((model, MODEL_NAME), (fallback_model, FALLBACK_MODEL_NAME)):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                chat = use_model.start_chat(history=chat_history)
                response = await chat.send_message_async(user_message)
                return response.text
            except google_exceptions.ServiceUnavailable:
                logger.warning(f"[{label}] 503 - yoğunluk hatası, deneme {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY * attempt)
                    continue
            except google_exceptions.ResourceExhausted:
                logger.warning(f"[{label}] 429 - kota/istek limiti aşıldı")
                return "⏳ Şu anda istek limitine ulaşıldı. Lütfen birkaç dakika sonra tekrar dener misin?"
            except google_exceptions.InvalidArgument as e:
                logger.error(f"[{label}] Geçersiz istek: {e}")
                return "⚠️ İsteğinde bir sorun var, farklı bir şekilde tekrar dener misin?"
            except google_exceptions.PermissionDenied:
                logger.error(f"[{label}] API key geçersiz veya yetkisiz")
                return "🔑 API anahtarıyla ilgili bir sorun var, lütfen yöneticiye bildirin."
            except Exception as e:
                logger.exception(f"[{label}] Beklenmeyen hata: {e}")
                break
        logger.info(f"[{label}] tüm denemeler tükendi, sıradaki modele geçiliyor.")

    return "😔 Şu anda yapay zeka servisine ulaşamıyorum (yoğunluk/bağlantı sorunu). Lütfen birazdan tekrar dener misin?"


def trim_history(user_id: int) -> None:
    history = user_histories[user_id]
    if len(history) > MAX_HISTORY_MESSAGES:
        user_histories[user_id] = history[-MAX_HISTORY_MESSAGES:]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_histories[update.effective_user.id] = []
    await update.message.reply_text(
        "👋 Merhaba! Ben senin yapay zeka asistanınım.\n\n"
        "İstediğini yazabilirsin, sana yardımcı olmaya çalışacağım.\n\n"
        "Komutlar:\n/reset — Konuşma geçmişini sıfırla\n/help — Yardım"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 Bu bot Google Gemini AI ile çalışır.\nBana istediğin soruyu yazman yeterli.\n\n/reset — hafızayı temizler\n/help — bu mesajı gösterir"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_histories[update.effective_user.id] = []
    await update.message.reply_text("🧹 Konuşma geçmişi temizlendi.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text

    if not user_message or not user_message.strip():
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    history = user_histories[user_id]

    try:
        reply_text = await generate_with_retry(history, user_message)
    except Exception as e:
        logger.exception(f"handle_message üst seviye hata: {e}")
        reply_text = "❌ Beklenmeyen bir hata oluştu, lütfen tekrar dener misin?"

    history.append({"role": "user", "parts": [user_message]})
    history.append({"role": "model", "parts": [reply_text]})
    trim_history(user_id)

    for i in range(0, len(reply_text), 4000):
        await update.message.reply_text(reply_text[i:i + 4000])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Bot genel hata yakaladı: {context.error}", exc_info=context.error)


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot başlatılıyor...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
