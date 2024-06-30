import os
import numpy as np
import streamlit as st
import cv2
import time
import random
import torch
import imutils
import math
from PIL import Image

# Function to convert PIL image to OpenCV format
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to convert OpenCV format to PIL image
def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))