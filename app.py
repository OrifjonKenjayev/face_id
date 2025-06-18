import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import base64
import pytz
from scipy.spatial import distance as dist
from flask import Flask, Response, render_template, request, jsonify
from threading import Lock, Thread
from decouple import config

# --- Configuration Constants ---
# In a real application, these paths should be configured securely
SHAPE_PREDICTOR_PATH = config('SHAPE_PREDICTOR_PATH', default='data/data_dlib/shape_predictor_68_face_landmarks.dat')
FACE_RECO_MODEL_PATH = config('FACE_RECO_MODEL_PATH', default='data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
FEATURES_CSV_PATH = config('FEATURES_CSV_PATH', default='data/features_all.csv')
DB_PATH = config('DB_PATH', default='attendance.db')

# --- Performance & Liveness Tuning ---
FACE_RECO_DISTANCE_THRESH = 0.4
CHALLENGE_TIMEOUT = 10  # Timeout for the blink challenge
VERIFIED_DISPLAY_DURATION = 3 # How long the "Verified" status shows

# --- Liveness Detection Constants (Blink Only) ---
EYE_AR_THRESH = 0.2       # Threshold for eye aspect ratio to detect a blink
EYE_AR_CONSEC_FRAMES = 1    # Number of consecutive frames the eye must be "closed"

# --- Global Objects & Locks ---
app = Flask(__name__, template_folder='templates')
db_lock = Lock()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
USER_TIMEZONE = pytz.timezone('Asia/Tashkent') # Set your local timezone

# --- State management ---
# This dictionary manages the state for a single user session.
user_state = {
    'liveness_challenge': {
        "face_id": None,
        "challenge_type": None,
        "start_time": None
    },
    'last_known_name': 'Unknown',
    'verified_until': 0,
    'blink_tracker': {'frames': 0, 'total_blinks': 0},
    'challenge_completed': False
}


# --- Database Functions ---
def init_database():
    """Creates the attendance table if it doesn't exist."""
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cursor = conn.cursor()
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                arrival_time TEXT,
                leaving_time TEXT,
                date TEXT NOT NULL,
                status TEXT,
                UNIQUE(name, date)
            );
            """
            cursor.execute(create_table_sql)
            conn.commit()
            conn.close()
        logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        exit()

# --- Core Frame Processing Class ---
class FrameProcessor:
    def __init__(self):
        """Loads all necessary dlib models and known face data."""
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
            self.face_reco_model = dlib.face_recognition_model_v1(FACE_RECO_MODEL_PATH)
        except RuntimeError as e:
            logging.error(f"Error loading Dlib models: {e}. Make sure the model files are in the correct path.")
            exit()

        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_known_faces()

    def load_known_faces(self):
        """Loads known face features and names from the CSV file."""
        if not os.path.exists(FEATURES_CSV_PATH):
            logging.warning(f"'{FEATURES_CSV_PATH}' not found. The application will run without any known faces.")
            return
        try:
            csv_rd = pd.read_csv(FEATURES_CSV_PATH, header=None)
            for i in range(csv_rd.shape[0]):
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(np.array(csv_rd.iloc[i, 1:], dtype=float))
            logging.info(f"Loaded {len(self.face_name_known_list)} known faces.")
        except Exception as e:
            logging.error(f"Failed to load or process '{FEATURES_CSV_PATH}': {e}")

    def update_attendance(self, name):
        """Logs arrival and updates leaving time in the database for a verified user."""
        now = datetime.datetime.now(USER_TIMEZONE)
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')
        try:
            with db_lock:
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
                entry = cursor.fetchone()
                
                if entry is None:
                    # First time this person is verified today, log their arrival.
                    cursor.execute(
                        "INSERT INTO attendance (name, arrival_time, date, status) VALUES (?, ?, ?, ?)",
                        (name, current_time, current_date, 'Present')
                    )
                    logging.info(f"Arrival marked for {name} at {current_time}")
                else:
                    # The person is already marked present; update their 'leaving_time' to the latest interaction.
                    cursor.execute(
                        "UPDATE attendance SET leaving_time = ? WHERE name = ? AND date = ?",
                        (current_time, name, current_date)
                    )
                conn.commit()
                conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database error during attendance update: {e}")

    @staticmethod
    def get_eye_aspect_ratio(eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        A = dist.euclidean(eye_points[1], eye_points[5]) # Vertical distance
        B = dist.euclidean(eye_points[2], eye_points[4]) # Vertical distance
        C = dist.euclidean(eye_points[0], eye_points[3]) # Horizontal distance
        ear = (A + B) / (2.0 * C)
        return ear

    def issue_new_challenge(self):
        """Issues a new blink challenge."""
        user_state['liveness_challenge'] = {
            "face_id": 0, # Placeholder for a unique ID if handling multiple faces
            "challenge_type": "blink", # Only blink challenge is supported now
            "start_time": time.time()
        }
        # Reset the blink tracker for the new challenge
        user_state['blink_tracker'] = {'frames': 0, 'total_blinks': 0}
        user_state['challenge_completed'] = False
        logging.info("Issued 'blink' challenge.")
        return user_state['liveness_challenge']

    def process_frame(self, frame):
        """Processes a single frame for face recognition and liveness detection."""
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = self.detector(rgb_frame, 0)
        
        result = {'status': 'no_face', 'name': 'Unknown', 'box': None, 'challenge': None}

        if not faces:
            user_state['liveness_challenge']['face_id'] = None # Reset challenge if face is lost
            return result

        face = faces[0] # Process only the first detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        result['box'] = [x, y, w, h]

        shape = self.predictor(rgb_frame, face)
        
        # --- Face Recognition ---
        features = np.array(self.face_reco_model.compute_face_descriptor(rgb_frame, shape))
        distances = [dist.euclidean(features, known) for known in self.face_features_known_list]
        name = "Unknown"
        if distances and np.min(distances) < FACE_RECO_DISTANCE_THRESH:
            name = self.face_name_known_list[np.argmin(distances)]
        
        user_state['last_known_name'] = name
        result['name'] = name

        # --- Liveness Logic ---
        is_verified = user_state.get('verified_until', 0) > time.time()
        
        if name == "Unknown":
            result['status'] = 'unknown'
            return result

        if is_verified:
            result['status'] = 'verified'
            self.update_attendance(name) # Keep updating leaving time
            return result

        # Issue a new challenge if one isn't active
        if not user_state['liveness_challenge'].get('challenge_type'):
            self.issue_new_challenge()
        
        challenge = user_state['liveness_challenge']
        result['challenge'] = challenge['challenge_type']
        result['status'] = 'challenge'

        # Check for challenge timeout
        if time.time() - challenge['start_time'] > CHALLENGE_TIMEOUT:
            logging.warning("Challenge timed out.")
            user_state['liveness_challenge'] = {} # Reset
            result['challenge'] = None
            result['status'] = 'timeout'
            return result
        
        # --- Check if blink challenge is passed ---
        action_passed = False
        
        # Get eye landmarks
        left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
        
        # Calculate EAR for both eyes
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Check for blink
        if ear < EYE_AR_THRESH:
            user_state['blink_tracker']['frames'] += 1
        else:
            if user_state['blink_tracker']['frames'] >= EYE_AR_CONSEC_FRAMES:
                user_state['blink_tracker']['total_blinks'] += 1
                logging.info(f"Blink completed! Total blinks: {user_state['blink_tracker']['total_blinks']}")
                action_passed = True
            user_state['blink_tracker']['frames'] = 0

        if action_passed and not user_state['challenge_completed']:
            logging.info(f"Challenge 'blink' passed by {name}.")
            user_state['verified_until'] = time.time() + VERIFIED_DISPLAY_DURATION
            user_state['challenge_completed'] = True
            self.update_attendance(name)
            user_state['liveness_challenge'] = {} # Reset challenge
            result['status'] = 'verified'
            result['challenge'] = None

        return result


# --- Global Frame Processor ---
processor = FrameProcessor()


# --- Flask App Routes ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_known_faces():
    if not os.path.exists(FEATURES_CSV_PATH): return []
    try:
        return pd.read_csv(FEATURES_CSV_PATH, header=None)[0].tolist()
    except Exception:
        return []

@app.route('/')
def home():
    """Home page with links to the dashboard and recognition app."""
    return render_template('home.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Displays the attendance dashboard."""
    all_names = get_all_known_faces()
    now_local = datetime.datetime.now(USER_TIMEZONE)
    selected_date_str = request.form.get('selected_date', now_local.strftime('%Y-%m-%d'))
    
    with db_lock:
        conn = get_db_connection()
        records = conn.execute("SELECT name, arrival_time, leaving_time FROM attendance WHERE date = ?", (selected_date_str,)).fetchall()
        conn.close()
    
    present_people = {r['name']: r for r in records}
    display_data = []
    for name in all_names:
        record = present_people.get(name)
        if record:
            display_data.append({'name': name, 'status': 'Present', 'arrival_time': record['arrival_time'] or '---', 'leaving_time': record['leaving_time'] or '---'})
        else:
            display_data.append({'name': name, 'status': 'Absent', 'arrival_time': '---', 'leaving_time': '---'})
    
    return render_template('index_dashboard.html', attendance_data=display_data, selected_date=selected_date_str, no_data=not display_data)

@app.route('/recognition')
def recognition_page():
    """Renders the face recognition/capture page."""
    return render_template('index_recognation.html')

@app.route('/recognition/process', methods=['POST'])
def process_client_frame():
    """Receives a frame from the client, processes it, and returns results."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    
    # Decode the base64 image
    try:
        header, encoded = data['image'].split(",", 1)
        binary_data = base64.b64decode(encoded)
    except (ValueError, TypeError) as e:
        return jsonify({'status': 'error', 'message': f'Invalid image data: {e}'}), 400
    
    nparr = np.frombuffer(binary_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Could not decode image'}), 400
        
    result = processor.process_frame(frame)
    
    return jsonify(result)

# --- Main Application Runner ---
def main():
    """Initializes the database and starts the Flask app."""
    init_database()
    # You might need to create a dummy 'features_all.csv' and the dlib models for it to run
    # For production, use a proper web server like Gunicorn or Waitress
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)

if __name__ == '__main__':
    main()
