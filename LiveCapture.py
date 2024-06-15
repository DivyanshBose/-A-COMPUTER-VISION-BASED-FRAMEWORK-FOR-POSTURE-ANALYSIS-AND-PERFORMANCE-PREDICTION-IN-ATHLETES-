import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import math as m
import time
from PIL import Image
import numpy as np


# Functions for posture detection
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree


def sendWarning():
    st.warning("Bad posture detected for over 3 minutes!")


# Page structure
PAGES = ["Login", "Sign Up", "Dashboard"]

# Dummy database
users_db = {"test_user": {"password": "password", "name": "Test User", "photo": "path/to/photo.jpg"}}

# User session state
if "page" not in st.session_state:
    st.session_state["page"] = "Login"

if "user" not in st.session_state:
    st.session_state["user"] = None


# Streamlit app structure
def main():
    if st.session_state["page"] == "Login":
        login_page()
    elif st.session_state["page"] == "Sign Up":
        signup_page()
    elif st.session_state["page"] == "Dashboard":
        dashboard_page()


def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users_db and users_db[username]["password"] == password:
            st.session_state["user"] = users_db[username]
            st.session_state["page"] = "Dashboard"
        else:
            st.error("Invalid credentials")
    if st.button("Sign Up"):
        st.session_state["page"] = "Sign Up"


def signup_page():
    st.title("Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    name = st.text_input("Full Name")
    photo = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])
    if st.button("Sign Up"):
        if username not in users_db:
            users_db[username] = {"password": password, "name": name, "photo": photo}
            st.session_state["user"] = users_db[username]
            st.session_state["page"] = "Dashboard"
        else:
            st.error("Username already exists")
    if st.button("Back to Login"):
        st.session_state["page"] = "Login"


def dashboard_page():
    user = st.session_state["user"]
    st.title(f"Welcome, {user['name']}")
    if user["photo"]:
        st.image(user["photo"], width=150)
    if st.button("Start Analysis"):
        st.session_state["analyzing"] = True
    if "analyzing" in st.session_state and st.session_state["analyzing"]:
        posture_detection()


def posture_detection():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    offset_data = {'Offset': [], 'Neck Inclination': [], 'Torso Inclination': [], 'Sport': []}
    good_frames = 0
    bad_frames = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (640, 480))
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            l_shldr_x, l_shldr_y = int(l_shldr.x * w), int(l_shldr.y * h)
            r_shldr_x, r_shldr_y = int(r_shldr.x * w), int(r_shldr.y * h)
            l_ear_x, l_ear_y = int(l_ear.x * w), int(l_ear.y * h)
            l_hip_x, l_hip_y = int(l_hip.x * w), int(l_hip.y * h)

            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            offset_data['Offset'].append(offset)
            offset_data['Neck Inclination'].append(neck_inclination)
            offset_data['Torso Inclination'].append(torso_inclination)
            offset_data['Sport'].append('Live')

            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1
                cv2.putText(img, f"Good Posture: Neck: {int(neck_inclination)}, Torso: {int(torso_inclination)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                good_frames = 0
                bad_frames += 1
                cv2.putText(img, f"Bad Posture: Neck: {int(neck_inclination)}, Torso: {int(torso_inclination)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            good_time = (1 / 30) * good_frames
            bad_time = (1 / 30) * bad_frames
            if bad_time > 180:
                sendWarning()

        frame_window.image(img_rgb)

        if st.button("Stop"):
            break

    cap.release()
    st.session_state["analyzing"] = False
    offset_df = pd.DataFrame(offset_data)
    st.download_button(
        label="Download dataset as CSV",
        data=offset_df.to_csv(index=False),
        file_name='dataset.csv',
        mime='text/csv'
    )


if __name__ == "__main__":
    main()
