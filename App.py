import os
import numpy as np
import streamlit as st
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from deepface import DeepFace
from collections import Counter
from utils import pil_to_cv2, cv2_to_pil  # Ensure these utility functions are correctly implemented

# Load models once when the application starts
session_state = st.session_state
if not hasattr(session_state, 'yolox'):
    session_state.yolox = YOLO("./models/yolov8x.pt")
if not hasattr(session_state, 'yolostand'):
    session_state.yolostand = YOLO("./models/sit_stand_model.pt")
if not hasattr(session_state, 'yolopg'):
    session_state.yolopg = YOLO("./models/PG_model.pt")
if not hasattr(session_state, 'yolopos'):
    session_state.yolopos = YOLO("./models/beerbox_model.pt")
if not hasattr(session_state, 'yolobeer'):
    session_state.yolobeer = YOLO("./models/beerbox_model.pt")

# Streamlit app
st.title('KTK - hackhcmc2024 - Challenge Statement 2 by HEINEKEN Vietnam')

st.sidebar.title('Select a Business Problem')
problem = st.sidebar.selectbox('Choose a problem to analyze', [
    'Problem 1: Count Beer Drinkers',
    'Problem 2: Detect Promotional Materials',
    'Problem 3: Evaluate Event Success',
    'Problem 4: Track Promotion Girls',
    'Problem 5: Grade Store Presence'
])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
submit = st.button('Analyze')

if uploaded_file is not None:
    # Convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)
    # Convert the PIL image to OpenCV format
    cv2_image = pil_to_cv2(image)

    ################################ PROBLEM 1 ##################################
    if problem == 'Problem 1' and submit:
        ### CODE HERE ###
        pass

    ################################ PROBLEM 2 ##################################
    if problem == 'Problem 2: Detect Promotional Materials' and submit:
        st.title('Results: Detect Promotional Materials')
        results = session_state.yolopos.predict(cv2_image, show=False, device=0)
        print(results[0].names)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        item_counts = Counter(clss)
        items_detected = [session_state.yolopos.names[int(cls)] for cls in clss if session_state.yolopos.names[int(cls)] not in ['Human', 'marketing staff']]

        # Count objects for each class, excluding 'person' and 'marketing staff'
        counts_per_class = {session_state.yolopos.names[int(cls)]: count for cls, count in item_counts.items() if session_state.yolopos.names[int(cls)] not in ['Human', 'marketing staff']}

        st.write("Detected items:")
        for item, count in counts_per_class.items():
            st.write(f"{item}: {count}")

        # Visualize results, excluding 'person' and 'marketing staff' classes
        annotator = Annotator(cv2_image.copy(), line_width=2, example=session_state.yolopos.names)
        for box, cls in zip(boxes, clss):
            if session_state.yolopos.names[int(cls)] not in ['Human', 'marketing staff']:
                annotator.box_label(box, color=colors(int(cls), True), label=session_state.yolopos.names[int(cls)])

        st.image(cv2_to_pil(annotator.result()))

    ################################ PROBLEM 3 ##################################
    if problem == 'Problem 3: Evaluate Event Success' and submit:
        st.title('Results: Evaluate Event Success')
        names = session_state.yolox.names

        results = session_state.yolox.predict(cv2_image, show=False, device=0, classes=0)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        cropped_images = []
        visualize_img = cv2_image.copy()
        idx = 0

        if boxes is not None:
            for box, cls in zip(boxes, clss):
                idx += 1
                class_name = names[int(cls)]
                cropped_images.append(cv2_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])

        annotator = Annotator(visualize_img, line_width=2, example=names)
        if boxes is not None:
            for box, cls in zip(boxes, clss):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

        st.image(cv2_to_pil(visualize_img))
        st.write(f"There are {idx} people in the image.")

        dominant_emotions = []

        for img in cropped_images:
            cv2.imwrite(f'./temp.jpg', img)
            try:
                demographies = DeepFace.analyze(
                    img_path='./temp.jpg',
                    detector_backend='yolov8',
                    align=True,
                    actions=['emotion']
                )
                dominant_emotion = demographies[0]['dominant_emotion']
                dominant_emotions.append(dominant_emotion)
                face_box = demographies[0]['region']
                x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                st.image(cv2_to_pil(img))
            except Exception as e:
                continue
        
        if len(dominant_emotions) > 0:
            emotion_counts = Counter(dominant_emotions)
            most_common_emotion = emotion_counts.most_common(1)[0][0]
            st.write(f"The most dominant emotion is '{most_common_emotion}'.")
        else:
            st.write("No dominant emotion detected.")

    ################################ PROBLEM 4 ##################################
    if problem == 'Problem 4: Track Promotion Girls' and submit:
        st.title('Results: Track Promotion Girls')
        names = session_state.yolox.names

        results = session_state.yolox.predict(cv2_image, show=False, device=0)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        cropped_images = []
        pg_count = 0

        visualize_img = cv2_image.copy()
        annotator = Annotator(visualize_img, line_width=2, example=names)

        if boxes is not None:
            for box, cls in zip(boxes, clss):
                class_name = names[int(cls)]
                if class_name == 'person':
                    cropped_images.append((cv2_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])], box))

        for image, box in cropped_images:
            cv2.imwrite(f'./temp.jpg', image)
            try:
                demographies = DeepFace.analyze(
                    img_path='./temp.jpg',
                    detector_backend='yolov8',
                    align=True,
                    actions=['gender']
                )
                dominant_gender = demographies[0]['dominant_gender']
                if dominant_gender == 'Woman':
                    ss_result = session_state.yolostand.predict(image)
                    predictions = ss_result[0].probs
                    predicted_class = predictions.top1
                    predicted_class_name = session_state.yolostand.names[predicted_class]
                    if int(predicted_class_name) == 1:
                        pg_result = session_state.yolopg.predict(image)
                        predictions = pg_result[0].probs
                        predicted_class = predictions.top1
                        predicted_class_name = session_state.yolopg.names[predicted_class]
                        if predicted_class_name == 'PromotionGirl':
                            pg_count += 1
                            annotator.box_label(box, color=colors(int(cls), True), label='PromotionGirl')
            except Exception as e:
                continue
        
        st.image(cv2_to_pil(annotator.result()))
        st.write(f'There are {pg_count} promotion girls.')

        if pg_count >= 2:
            st.write("There are more than 2 promotion girls in the image.")
        else:
            st.write("There are less than 2 promotion girls in the image.")

    ################################ PROBLEM 5 ##################################
    if problem == 'Problem 5: Grade Store Presence' and submit:
        st.title('Results: Grade Store Presence')
        results = session_state.yolopos.predict(cv2_image, show=False, device=0)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        heineken_billboard_count = 0
        heineken_fridge_count = 0
        heineken_keg_count = 0

        heineken_billboard_classes = [15]  # Class index for 'banner- Heineken'
        heineken_fridge_classes = [11]  # Class index for 'Fridge'
        heineken_keg_classes = [2, 7]  # Class indices for 'Beer box- Heineken' and 'Beer- Heineken'

        for cls in clss:
            class_name = session_state.yolopos.names[int(cls)]
            if int(cls) in heineken_billboard_classes:
                heineken_billboard_count += 1
            elif int(cls) in heineken_fridge_classes:
                heineken_fridge_count += 1
            elif int(cls) in heineken_keg_classes:
                heineken_keg_count += 1

        st.write(f"Heineken Billboards: {heineken_billboard_count}")
        st.write(f"Heineken Fridges: {heineken_fridge_count}")
        st.write(f"Heineken Kegs: {heineken_keg_count}")

        if heineken_billboard_count >= 1 and heineken_fridge_count >= 1 and heineken_keg_count >= 10:
            st.write("Heineken's presence is well established in this store.")
        else:
            st.write("Heineken's presence is not sufficient in this store.")

        # Visualize results, excluding 'Human' and 'marketing staff' classes
        annotator = Annotator(cv2_image.copy(), line_width=2, example=session_state.yolopos.names)
        for box, cls in zip(boxes, clss):
            if session_state.yolopos.names[int(cls)] not in ['Human', 'marketing staff']:
                annotator.box_label(box, color=colors(int(cls), True), label=session_state.yolopos.names[int(cls)])

        st.image(cv2_to_pil(annotator.result()))