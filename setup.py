from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='stablediffusion-telegram-bot',
    version='1.0',
    description='Telegram bot for Stable Diffusion neural network',
    author='DirectX',
    packages=['app'],
    install_requires=required,
    project_urls={
        "Source Code": "https://github.com/DirectX/stablediffusion-telegram-bot"
    },
)
