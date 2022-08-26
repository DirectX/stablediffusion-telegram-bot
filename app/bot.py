from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
from .translator import Translator
from .stablediffusion import StableDiffusion

ru_en_translator = Translator('ru', 'en')

def run(telegram_bot_token: str, stable_diffusion: StableDiffusion) -> None:
    print('Starting telegram bot...')

    application = ApplicationBuilder().token(telegram_bot_token).build()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(
            chat_id = update.effective_chat.id,
            text="Бот для генерации изображений нейросетью Stable Diffusion\n\nНапишите фразу на русском или английском языке и бот сгенерирует изображение, используя нейросеть Stable Diffusion"
        )

    async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
        translated_text = ru_en_translator.translate(update.message.text)
        filepaths = stable_diffusion.text2img(translated_text)
        
        for filepath in filepaths:
            await context.bot.send_photo(update.effective_chat.id, photo=open(filepath, 'rb'), caption=translated_text)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_text)
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    translate_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), translate)
    application.add_handler(translate_handler)

    print('Waiting for messages...')
    application.run_polling()
