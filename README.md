APP DEMO:
https://www.figma.com/design/PGCYRKoadKXW1vB9SrC3eA/lamah?node-id=8-2&t=UR3xLEVw62j6Tjxu-1


Required Libraries:<br>
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
