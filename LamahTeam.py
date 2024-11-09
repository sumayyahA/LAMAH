from ibm_watsonx_ai.foundation_models import Model
import cv2
from ultralytics import YOLO
import streamlit as st
import json
import re
import arabic_reshaper 
from bidi.algorithm import get_display
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from gtts import gTTS

# إعدادات الاتصال بموديل علّام
def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": "ZUJc1B39YUMlr-k1AM4aYS49KnXKjA_q2T6reGHQ0qew"
    }

def initialize_model():
    model_id = "sdaia/allam-1-13b-instruct"
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 900,
        "repetition_penalty": 1,
        "temperature": 0.8,
        "stop_sequences": ["<s>"],
        "seed": 123
    }
    project_id = "22620ae3-9605-48e4-9c0d-2784e5f7c638"
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id
    )
    return model

ALLAM_model = initialize_model()

def get_Allam_response(object):
    prompt_input = f"You are an Arabic language teacher for a child aged 4-8 years, who is exploring the world around them."
    question = f"""Translate the Word:"{object}" into Arabic, and Generate exactly 3 sentences of 5 words each. Present the result in the following format: [word_in_arabic, sentence_1, sentence_2, sentence_3] with classes WORD, SEN_1, SEN_2, SEN_3 in JSON format. ADD DIACRITICAL MARKS."""
    formattedQuestion = f"""<s> [INST] {question} [/INST]"""
    prompt = f"""{prompt_input} {formattedQuestion}"""
    generated_response = ALLAM_model.generate_text(prompt=prompt, guardrails=False)
    print(f"ALLAM: {generated_response}")
    return generated_response

def extract_list(text):
    match = re.search(r'\[(.*?)\]', text)
    if match:
        list_content = match.group(1)
        return [item.strip().replace('"', '') for item in list_content.split(',')]
    return None

# Set background image and row-based layout
st.markdown("""
    <style>
        /* Full-page background */
        .stApp {
            background-image: url('https://drive.google.com/uc?export=view&id=1K0BIWRv8X-NM48JOoOY0Oqc9CQfsHXL7');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Row styling */
        .row {
            width: 100%;
            max-width: 600px;
            margin: 20px auto;
            background-color: #ffffffcc;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .camera-row {
            background-color: #933793;
            border-radius: 15px;
            padding: 20px;
        }

        /* Text and audio button styling */
        .audio-button {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #D4A5D4;
            color: white;
            font-weight: bold;
            cursor: pointer;
            display: inline-block;
            margin-left: 10px;
        }
        
        .text-audio {
            font-size: 20px;
            color: #333333;
            display: inline-block;
        }

    </style>
""", unsafe_allow_html=True)

# إدخال اسم المستخدم
user_name = st.text_input("أدخل اسمك:", "أحمد")
st.markdown(f"<div class='title'> مرحبًا {user_name}، ماذا ستكتشف اليوم؟ </div>", unsafe_allow_html=True)

# Title Row
#st.markdown("<div class='row'><h2>مرحبًا أحمد، ماذا ستكتشفُ اليوم؟</h2></div>", unsafe_allow_html=True)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Camera Row
st.markdown("<div class='row camera-row'>", unsafe_allow_html=True)
camera_input = st.camera_input("التقط صورة")  # Camera component inside the row
st.markdown("</div>", unsafe_allow_html=True)

# Process the captured image
if camera_input is not None:
    cv2_img = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    results = model(cv2_img)
    largest_object_label = ""
    max_area = 0
    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_object_class_id = int(obj.cls)
                largest_object_label = model.names[largest_object_class_id]

    # Display the recognized word and its audio in rows
    if largest_object_label:
        response = get_Allam_response(largest_object_label)
        json_part = extract_list(response)

        if json_part:
            allam_word = re.sub(r'[^\u0600-\u06FF\s]', '', json_part[0].replace("_", ""))
            allam_sen1 = re.sub(r'[^\u0600-\u06FF\s]', '', json_part[1].replace("_", ""))
            allam_sen2 = re.sub(r'[^\u0600-\u06FF\s]', '', json_part[2].replace("_", ""))
            allam_sen3 = re.sub(r'[^\u0600-\u06FF\s]', '', json_part[3].replace("_", ""))

            # Display word with audio button in a row
            st.markdown(f"""
                <div class='row'>
                    <span class='text-audio'><b>الكلمة:</b> {allam_word}</span>
                    <a href='word_audio.wav' class='audio-button' download>🔊 استمع إلى الكلمة</a>
                </div>
            """, unsafe_allow_html=True)
            text_to_speech1 = gTTS(allam_word, lang='ar', tld='com')
            text_to_speech1.save('word_audio.wav')
            st.audio('word_audio.wav', format="audio/wav")

            # Display each sentence with audio button in a row
            # Loop through sentences and place text and audio button side by side
            for idx, sentence in enumerate([allam_sen1, allam_sen2, allam_sen3], start=1):
                # Create the audio file for each sentence
                text_to_speech = gTTS(sentence, lang='ar', tld='com')
                text_to_speech.save(f'sen{idx}_audio.wav')

                # Display the sentence and audio button side by side
                st.markdown(f"""
                    <div class='row' style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                        <div class='text-audio' style="flex: 1; text-align: right;">
                            <b>الجملة {idx}:</b> {sentence}
                        </div>
                        <a href='sen{idx}_audio.wav' class='audio-button' style="flex: 0;" download>🔊 استمع إلى الجملة {idx}</a>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display audio player
                st.audio(f'sen{idx}_audio.wav', format="audio/wav")