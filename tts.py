from deepgram import DeepgramClient, SpeakOptions
import pygame
DEEPGRAM_API_KEY = "6c7d334900cf23f6165912ee9deba71925dc2ea6"

FILENAME = "audio.mp3"
def textToAudio(text):
    TEXT = {
    "text": text
}
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        options = SpeakOptions(
            model="aura-asteria-en",
        )
        response = deepgram.speak.v("1").save(FILENAME, TEXT, options)
        print(response.to_json(indent=4))
        # return FILENAME
    except Exception as e:
        print(f"Exception: {e}")
      

