












# import dlib
# import numpy as np
# import cv2
# import os
# import pandas as pd
# import time
# import logging
# import sqlite3
# import datetime
# from scipy.spatial import distance as dist

# # --- Configuration Constants ---
# # Path to dlib's pre-trained models
# SHAPE_PREDICTOR_PATH = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
# FACE_RECO_MODEL_PATH = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

# # Path to the CSV file with known face features
# FEATURES_CSV_PATH = "data/features_all.csv"

# # Database file
# DB_PATH = "attendance.db"

# # Liveness Detection Thresholds
# EYE_AR_THRESH = 0.25  # Threshold for eye aspect ratio to indicate a blink
# EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames the eye must be below the threshold

# # Recognition threshold
# FACE_RECO_DISTANCE_THRESH = 0.4

# class FaceRecognizer:
#     def __init__(self):
#         # --- Initialize Dlib Models ---
#         self.detector = dlib.get_frontal_face_detector()
#         try:
#             self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
#             self.face_reco_model = dlib.face_recognition_model_v1(FACE_RECO_MODEL_PATH)
#         except RuntimeError as e:
#             logging.error(f"Error loading Dlib models: {e}")
#             logging.error(f"Please make sure the files '{SHAPE_PREDICTOR_PATH}' and '{FACE_RECO_MODEL_PATH}' exist.")
#             exit()

#         self.font = cv2.FONT_HERSHEY_SIMPLEX

#         # --- Frame processing variables ---
#         self.frame_cnt = 0
#         self.fps = 0
#         self.start_time = time.time()

#         # --- Known Faces Data ---
#         self.face_features_known_list = []
#         self.face_name_known_list = []

#         # --- Tracking Liveness ---
#         self.blink_counters = {}  # {person_name: count}
#         self.is_live = {}         # {person_name: boolean}

#         # --- Initialize Database ---
#         self.init_database()

#     def init_database(self):
#         """Initializes the SQLite database and creates the attendance table if it doesn't exist."""
#         try:
#             conn = sqlite3.connect(DB_PATH)
#             cursor = conn.cursor()
#             # Added leaving_time and status columns
#             create_table_sql = """
#             CREATE TABLE IF NOT EXISTS attendance (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT NOT NULL,
#                 arrival_time TEXT,
#                 leaving_time TEXT,
#                 date TEXT NOT NULL,
#                 status TEXT,
#                 UNIQUE(name, date)
#             );
#             """
#             cursor.execute(create_table_sql)
#             conn.commit()
#             conn.close()
#             logging.info("Database initialized successfully.")
#         except sqlite3.Error as e:
#             logging.error(f"Database error: {e}")
#             exit()

#     def get_face_database(self):
#         """Loads known faces from the CSV file."""
#         if not os.path.exists(FEATURES_CSV_PATH):
#             logging.warning(f"'{FEATURES_CSV_PATH}' not found!")
#             logging.warning("Please run scripts to generate face features CSV first.")
#             return False

#         try:
#             csv_rd = pd.read_csv(FEATURES_CSV_PATH, header=None)
#             for i in range(csv_rd.shape[0]):
#                 name = csv_rd.iloc[i][0]
#                 features = np.array(csv_rd.iloc[i, 1:], dtype=float)
#                 self.face_name_known_list.append(name)
#                 self.face_features_known_list.append(features)
#                 # Initialize liveness tracking for each known person
#                 self.blink_counters[name] = 0
#                 self.is_live[name] = False
#             logging.info(f"Loaded {len(self.face_features_known_list)} faces from the database.")
#             return True
#         except Exception as e:
#             logging.error(f"Failed to load or parse '{FEATURES_CSV_PATH}': {e}")
#             return False

#     @staticmethod
#     def eye_aspect_ratio(eye):
#         """Calculates the eye aspect ratio (EAR)."""
#         # Vertical eye landmarks
#         A = dist.euclidean(eye[1], eye[5])
#         B = dist.euclidean(eye[2], eye[4])
#         # Horizontal eye landmark
#         C = dist.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear

#     def check_liveness(self, shape, name):
#         """Checks for eye blinks to determine liveness."""
#         # Extract left and right eye coordinates from the 68 landmarks
#         (lStart, lEnd) = (42, 48)
#         (rStart, rEnd) = (36, 42)
#         left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lStart, lEnd)])
#         right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(rStart, rEnd)])

#         left_ear = self.eye_aspect_ratio(left_eye)
#         right_ear = self.eye_aspect_ratio(right_eye)
#         ear = (left_ear + right_ear) / 2.0

#         if ear < EYE_AR_THRESH:
#             self.blink_counters[name] += 1
#         else:
#             if self.blink_counters[name] >= EYE_AR_CONSEC_FRAMES:
#                 self.is_live[name] = True  # Liveness confirmed
#             self.blink_counters[name] = 0

#         return self.is_live[name]

#     def update_attendance(self, name):
#         """Records arrival and updates leaving time in the database."""
#         current_date = datetime.datetime.now().strftime('%Y-%m-%d')
#         current_time = datetime.datetime.now().strftime('%H:%M:%S')

#         try:
#             conn = sqlite3.connect(DB_PATH)
#             cursor = conn.cursor()

#             cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
#             entry = cursor.fetchone()

#             if entry is None:
#                 # First time seen today, record arrival
#                 cursor.execute(
#                     "INSERT INTO attendance (name, arrival_time, date, status) VALUES (?, ?, ?, ?)",
#                     (name, current_time, current_date, 'Present')
#                 )
#                 logging.info(f"Arrival marked for {name} at {current_time}")
#             else:
#                 # Already arrived, update leaving time
#                 cursor.execute(
#                     "UPDATE attendance SET leaving_time = ? WHERE name = ? AND date = ?",
#                     (current_time, name, current_date)
#                 )
#             conn.commit()
#             conn.close()
#         except sqlite3.Error as e:
#             logging.error(f"Database error during attendance update: {e}")

#     def process_frame(self, img_rd):
#         """Processes a single frame for face detection and recognition."""
#         gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
#         faces = self.detector(gray, 0)

#         self.frame_cnt += 1
#         self.update_fps()

#         for face in faces:
#             shape = self.predictor(img_rd, face)
#             face_features = self.face_reco_model.compute_face_descriptor(img_rd, shape)
#             face_features_np = np.array(face_features)

#             distances = [dist.euclidean(face_features_np, known_face) for known_face in self.face_features_known_list]
            
#             if distances:
#                 min_dist_idx = np.argmin(distances)
#                 min_dist = distances[min_dist_idx]
                
#                 name = "Unknown"
#                 color = (0, 0, 255) # Red for unknown
                
#                 if min_dist < FACE_RECO_DISTANCE_THRESH:
#                     name = self.face_name_known_list[min_dist_idx]
                    
#                     # Check for liveness only if the face is recognized
#                     is_live = self.check_liveness(shape, name)

#                     if is_live:
#                         self.update_attendance(name)
#                         color = (0, 255, 0)  # Green for live, recognized person
#                         liveness_text = "Live"
#                     else:
#                         color = (0, 255, 255) # Yellow for not yet live
#                         liveness_text = "Checking Liveness..."
                    
#                     # Draw bounding box and name
#                     (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
#                     cv2.rectangle(img_rd, (x, y), (x + w, y + h), color, 2)
#                     cv2.putText(img_rd, f"{name} ({liveness_text})", (x, y - 10), self.font, 0.7, color, 2)

#                 else: # Still draw for unknown faces
#                     (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
#                     cv2.rectangle(img_rd, (x, y), (x + w, y + h), color, 2)
#                     cv2.putText(img_rd, name, (x, y - 10), self.font, 0.7, color, 2)


#         self.draw_info(img_rd, len(faces))
#         return img_rd

#     def update_fps(self):
#         """Calculates and updates the FPS."""
#         now = time.time()
#         if now - self.start_time >= 1:
#             self.fps = self.frame_cnt / (now - self.start_time)
#             self.frame_cnt = 0
#             self.start_time = now

#     def draw_info(self, img_rd, face_count):
#         """Draws information text on the frame."""
#         cv2.putText(img_rd, f"FPS: {self.fps:.2f}", (20, 50), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.putText(img_rd, f"Faces detected: {face_count}", (20, 90), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.putText(img_rd, "Press 'q' to quit", (20, img_rd.shape[0] - 20), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


#     def run(self):
#         """Main loop to capture from camera and process."""
#         if not self.get_face_database():
#             return
        
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             logging.error("Cannot open camera.")
#             return

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.error("Can't receive frame (stream end?). Exiting ...")
#                 break

#             processed_frame = self.process_frame(frame)
#             cv2.imshow('Face Recognition Attendance', processed_frame)

#             if cv2.waitKey(1) == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()


# def main():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     recognizer = FaceRecognizer()
#     recognizer.run()


# if __name__ == '__main__':
#     main()













import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
from scipy.spatial import distance as dist
from flask import Flask, Response, render_template
from threading import Lock

# --- Configuration Constants ---
SHAPE_PREDICTOR_PATH = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECO_MODEL_PATH = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
FEATURES_CSV_PATH = "data/features_all.csv"
DB_PATH = "attendance.db"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
FACE_RECO_DISTANCE_THRESH = 0.4

# Initialize Flask app
app = Flask(__name__)
camera_lock = Lock()

class FaceRecognizer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
            self.face_reco_model = dlib.face_recognition_model_v1(FACE_RECO_MODEL_PATH)
        except RuntimeError as e:
            logging.error(f"Error loading Dlib models: {e}")
            logging.error(f"Please make sure the files '{SHAPE_PREDICTOR_PATH}' and '{FACE_RECO_MODEL_PATH}' exist.")
            exit()

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_cnt = 0
        self.fps = 0
        self.start_time = time.time()
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.blink_counters = {}
        self.is_live = {}
        self.cap = None
        self.init_database()

    def init_database(self):
        try:
            conn = sqlite3.connect(DB_PATH)
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
            logging.error(f"Database error: {e}")
            exit()

    def get_face_database(self):
        if not os.path.exists(FEATURES_CSV_PATH):
            logging.warning(f"'{FEATURES_CSV_PATH}' not found!")
            logging.warning("Please run scripts to generate face features CSV first.")
            return False

        try:
            csv_rd = pd.read_csv(FEATURES_CSV_PATH, header=None)
            for i in range(csv_rd.shape[0]):
                name = csv_rd.iloc[i][0]
                features = np.array(csv_rd.iloc[i, 1:], dtype=float)
                self.face_name_known_list.append(name)
                self.face_features_known_list.append(features)
                self.blink_counters[name] = 0
                self.is_live[name] = False
            logging.info(f"Loaded {len(self.face_features_known_list)} faces from the database.")
            return True
        except Exception as e:
            logging.error(f"Failed to load or parse '{FEATURES_CSV_PATH}': {e}")
            return False

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def check_liveness(self, shape, name):
        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)
        left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(lStart, lEnd)])
        right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(rStart, rEnd)])

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            self.blink_counters[name] += 1
        else:
            if self.blink_counters[name] >= EYE_AR_CONSEC_FRAMES:
                self.is_live[name] = True
            self.blink_counters[name] = 0

        return self.is_live[name]

    def update_attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
            entry = cursor.fetchone()

            if entry is None:
                cursor.execute(
                    "INSERT INTO attendance (name, arrival_time, date, status) VALUES (?, ?, ?, ?)",
                    (name, current_time, current_date, 'Present')
                )
                logging.info(f"Arrival marked for {name} at {current_time}")
            else:
                cursor.execute(
                    "UPDATE attendance SET leaving_time = ? WHERE name = ? AND date = ?",
                    (current_time, name, current_date)
                )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Database error during attendance update: {e}")

    def process_frame(self, img_rd):
        gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        self.frame_cnt += 1
        self.update_fps()

        for face in faces:
            shape = self.predictor(img_rd, face)
            face_features = self.face_reco_model.compute_face_descriptor(img_rd, shape)
            face_features_np = np.array(face_features)

            distances = [dist.euclidean(face_features_np, known_face) for known_face in self.face_features_known_list]
            
            if distances:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                name = "Unknown"
                color = (0, 0, 255)
                
                if min_dist < FACE_RECO_DISTANCE_THRESH:
                    name = self.face_name_known_list[min_dist_idx]
                    is_live = self.check_liveness(shape, name)

                    if is_live:
                        self.update_attendance(name)
                        color = (0, 255, 0)
                        liveness_text = "Live"
                    else:
                        color = (0, 255, 255)
                        liveness_text = "Checking Liveness..."
                    
                    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                    cv2.rectangle(img_rd, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img_rd, f"{name} ({liveness_text})", (x, y - 10), self.font, 0.7, color, 2)
                else:
                    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                    cv2.rectangle(img_rd, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img_rd, name, (x, y - 10), self.font, 0.7, color, 2)

        self.draw_info(img_rd, len(faces))
        return img_rd

    def update_fps(self):
        now = time.time()
        if now - self.start_time >= 1:
            self.fps = self.frame_cnt / (now - self.start_time)
            self.frame_cnt = 0
            self.start_time = now

    def draw_info(self, img_rd, face_count):
        cv2.putText(img_rd, f"FPS: {self.fps:.2f}", (20, 50), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_rd, f"Faces detected: {face_count}", (20, 90), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def generate_frames(self):
        if not self.get_face_database():
            logging.error("Failed to load face database. Exiting stream.")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
            return

        with camera_lock:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("Cannot open camera.")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.error("Can't receive frame (stream end?). Exiting ...")
                        break

                    processed_frame = self.process_frame(frame)
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if not ret:
                        logging.error("Failed to encode frame.")
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            finally:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

# Initialize recognizer
recognizer = FaceRecognizer()

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognizer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        app.run(host='0.0.0.0', port=5002, threaded=True, debug=False)
    finally:
        recognizer.release()

if __name__ == '__main__':
    main()