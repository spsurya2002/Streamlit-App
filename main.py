import streamlit as st
from groq import Groq
from PIL import Image, ImageGrab
import google.generativeai as genai
import cv2
import pyperclip
import speech_recognition as sr
from faster_whisper import WhisperModel
import pyaudio
import re
import os
import time
import wave
from dotenv import load_dotenv
from threading import Thread, Event
import mss  # New import for cross-platform screenshots
import pygame

load_dotenv()

# API configurations
groq_client_api_key = os.getenv("groq_client_api_key")
genai_api_key = os.getenv("genai_api_key")

groq_client = Groq(api_key=groq_client_api_key)
genai.configure(api_key=genai_api_key)

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}
safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                generation_config=generation_config,
                                safety_settings=safety_settings)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and factual '
    'response possible, carefully considering all previous generated text in your response before adding new tokens '
    'to the response...'
)
convo = [{'role': 'system', 'content': sys_msg}]

# Whisper and Speech Configuration
num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

# Streamlit UI
st.title('Voice AI Assistant with Multimedia Inputs')

# Input field for user prompt
prompt = st.text_input("Enter your prompt:")

# Alternative API communication functions
def groq_prompt(prompt, img_context=None):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user\'s prompt...'
    )
    function_convo = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

# Alternative for taking a screenshot using MSS
def take_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        screenshot_path = 'screenshot.png'
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)
        return screenshot_path

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)
    if not web_cam.isOpened():
        st.error("Error: Camera did not open successfully.")
        return None
    ret, frame = web_cam.read()
    path = 'webcam.jpg'
    cv2.imwrite(path, frame)
    web_cam.release()
    return path

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    return clipboard_content if isinstance(clipboard_content, str) else None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context'
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant'
        'to the user. Instead, take the user prompt input and try to extract all meaning from the photo'
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text


from  tts import textToAudio
def play():
    pygame.mixer.init()
    pygame.mixer.music.load('audio.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    return ''.join(segment.text for segment in segments)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    return match.group(1).strip() if match else None
if st.button("ss"):
    st.write("Taking screenshot...")
    screenshot_path = take_screenshot()
    image = Image.open(screenshot_path)
    st.image(image, caption='Screenshot', use_column_width=True)
    visual_context = vision_prompt(prompt, screenshot_path)
    response = groq_prompt(prompt, visual_context)
    st.write(f"AI Response: {response}")
    st.write("Speaking response...")
    textToAudio(response)
    play()
if st.button("webcam"):
    st.write("Taking webcam...")
    path = web_cam_capture()
    image = Image.open(path)
    st.image(image, caption='webcam', use_column_width=True)
    visual_context = vision_prompt(prompt, path)
    response = groq_prompt(prompt, visual_context)
    st.write(f"AI Response: {response}")
    
    st.write("Speaking response...")
    textToAudio(response)
    play()
# Handling user actions in Streamlit
if st.button("Process"):
    call = function_call(prompt)

    if "take screenshot" in call:
        st.write("Taking screenshot...")
        screenshot_path = take_screenshot()
        visual_context = vision_prompt(prompt, screenshot_path)
        st.image(screenshot_path)

    elif "capture webcam" in call:
        st.write("Capturing webcam...")
        webcam_path = web_cam_capture()
        if webcam_path:
            visual_context = vision_prompt(prompt, webcam_path)
            st.image(webcam_path)

    elif "extract clipboard" in call:
        st.write("Extracting clipboard content...")
        clipboard_text = get_clipboard_text()
        if clipboard_text:
            st.write(f"Clipboard Content: {clipboard_text}")
            prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {clipboard_text}'
        visual_context = None
    else:
        visual_context = None

    # Get AI response
    response = groq_prompt(prompt, visual_context)
    st.write(f"AI Response: {response}")
    
    st.write("Speaking response...")
    textToAudio(response)
    play()
    
    


