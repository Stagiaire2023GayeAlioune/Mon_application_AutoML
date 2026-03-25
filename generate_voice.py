from gtts import gTTS
from config import MARKETING_MESSAGE
import os

def generate_voice():
    os.makedirs("output", exist_ok=True)
    tts = gTTS(text=MARKETING_MESSAGE, lang="fr")
    tts.save("output/voice.mp3")
    return "output/voice.mp3"
