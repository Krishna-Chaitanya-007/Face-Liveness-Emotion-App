import os
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import random
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder='.')

# --- Model Loading ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(PREDICTOR_PATH):
    logging.error(f"FATAL ERROR: Dlib landmark model not found at '{PREDICTOR_PATH}'")
    logging.error("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    logging.info("Dlib models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dlib models: {e}")
    exit()

# --- Liveness Challenge Logic ---
CHALLENGES = ['Blink', 'Smile', 'Look Left', 'Look Right']

def get_eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose_ratio(landmarks):
    nose_tip = landmarks.part(33)
    left_edge = landmarks.part(0)
    right_edge = landmarks.part(16)
    dist_left = np.linalg.norm([nose_tip.x - left_edge.x, nose_tip.y - left_edge.y])
    dist_right = np.linalg.norm([nose_tip.x - right_edge.x, nose_tip.y - right_edge.y])
    if dist_left == 0: return float('inf')
    return dist_right / dist_left

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_challenge')
def get_challenge():
    challenge = random.choice(CHALLENGES)
    logging.info(f"Issuing new challenge: {challenge}")
    return jsonify({'challenge': challenge})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    image_data_url = data.get('image')
    challenge = data.get('challenge')

    if not image_data_url or not challenge:
        return jsonify({'success': False, 'reason': 'Missing data'}), 400

    try:
        img_data = base64.b64decode(image_data_url.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return jsonify({'success': False, 'reason': 'Invalid image format'}), 400

    rects = detector(gray, 0)
    if len(rects) == 0:
        return jsonify({'success': False, 'reason': 'No face detected'})

    rect = rects[0]
    landmarks = predictor(gray, rect)
    
    success = False
    if challenge == 'Blink':
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        ear = (get_eye_aspect_ratio(left_eye) + get_eye_aspect_ratio(right_eye)) / 2.0
        if ear < 0.2: success = True
    elif challenge == 'Smile':
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='dlib')
            if result and result[0]['dominant_emotion'] == 'happy': success = True
        except: success = False
    elif challenge == 'Look Left':
        if get_head_pose_ratio(landmarks) > 1.8: success = True
    elif challenge == 'Look Right':
        if get_head_pose_ratio(landmarks) < 0.55: success = True

    if success: logging.info(f"Challenge '{challenge}' PASSED.")
    return jsonify({'success': success})

# --- NEW: Emotion Analysis Endpoint ---
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """
    Analyzes a single frame for emotion without a liveness check.
    This is called continuously after the user is verified.
    """
    image_data_url = request.json.get('image')
    if not image_data_url:
        return jsonify({'error': 'No image data'}), 400

    try:
        img_data = base64.b64decode(image_data_url.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400
    
    rects = detector(gray, 0)
    if len(rects) == 0:
        return jsonify({'emotion': 'N/A', 'box': []})

    rect = rects[0]
    box = [rect.left(), rect.top(), rect.right(), rect.bottom()]
    
    emotion = "N/A"
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='dlib')
        if result:
            emotion = result[0]['dominant_emotion'].capitalize()
    except Exception as e:
        logging.warning(f"DeepFace failed during continuous analysis: {e}")

    return jsonify({'emotion': emotion, 'box': box})


if __name__ == '__main__':
    app.run(debug=True, threaded=False)

