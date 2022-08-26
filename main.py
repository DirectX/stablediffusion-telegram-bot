import os
from dotenv import load_dotenv
import app

if __name__ == "__main__":
    load_dotenv()
    
    stable_diffusion = app.StableDiffusion(os.getenv('STABLE_DIFFUSION_PATH'))
    app.run(os.getenv('TELEGRAM_BOT_TOKEN'), stable_diffusion)