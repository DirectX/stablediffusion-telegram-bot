from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
from .translator import Translator

ru_en_ranslator = Translator('ru', 'en')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id = update.effective_chat.id,
        text="Бот для генерации изображений нейросетью Stable Diffusion"
    )

async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    translated_text = ru_en_ranslator.translate(update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_text)

def run(telegram_bot_token: str) -> None:
    print('Starting telegram bot...')

    application = ApplicationBuilder().token(telegram_bot_token).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    translate_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), translate)
    application.add_handler(translate_handler)

    application.run_polling()