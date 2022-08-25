import os
from dotenv import load_dotenv
import app

if __name__ == "__main__":
    load_dotenv()

    app.run(os.getenv('TELEGRAM_BOT_TOKEN'))