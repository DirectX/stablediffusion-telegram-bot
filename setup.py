from setuptools import setup

setup(
    name='stablediffusion-telegram-bot',
    version='1.0',
    description='Telegram bot for Stable Diffusion neural network',
    author='DirectX',
    packages=['bot'],
    install_requires=['argostranslate', 'python-dotenv', 'python-telegram-bot'],
    project_urls={
        "Source Code": "https://github.com/DirectX/stablediffusion-telegram-bot"
    },
)
