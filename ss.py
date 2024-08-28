import streamlit as st
from PIL import Image, ImageGrab
import mss  # New import for cross-platform screenshots


def take_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        screenshot_path = 'screenshot.png'
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)
        return screenshot_path
    
if st.button("ss"):
    st.write("Taking screenshot...")
    screenshot_path = take_screenshot()
    image = Image.open(screenshot_path)
    st.image(image, caption='Screenshot', use_column_width=True)

