import os
import cv2
import torch
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Initialize Flask application and enable CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

# Set up file upload and processing folders
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}

# Configure Flask settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Limit upload size to 500 MB

# Ensure upload and processing folders exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

# Load YOLO model for object detection, with error handling
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Check if the uploaded file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert WebM files to MP4 format using ffmpeg
def convert_to_mp4(webm_path, mp4_path):
    ffmpeg_path = r'C:\path\to\ffmpeg\bin\ffmpeg.exe'  # Update ffmpeg path
    command = [ffmpeg_path, '-i', webm_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-y', mp4_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Handle file upload requests
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'Video uploaded successfully', 'file': filename}), 200
    return jsonify({'message': 'Invalid file format'}), 400

# Process uploaded video file with YOLO and color detection for team tracking
@app.route('/static/uploads/<filename>', methods=['GET'])
def process_video(filename):
    try:
        # Check and convert WebM files to MP4
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if filename.endswith('.webm'):
            converted_filename = filename.replace('.webm', '.mp4')
            converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
            convert_to_mp4(video_path, converted_video_path)
            filename = converted_filename
            video_path = converted_video_path

        processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cap = cv2.VideoCapture(video_path)

        # Error if video can't be opened
        if not cap.isOpened():
            return jsonify({'message': 'Error opening video file'}), 400

        # Set up video writer for processed output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

        def detect_color(frame, lower_bound, upper_bound):
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            return cv2.countNonZero(mask) > 500
        
        # Define color bounds in HSV for white, red, and black detection
        WHITE_LOWER = (0, 0, 168)
        WHITE_UPPER = (172, 111, 255)
        RED_LOWER = (0, 70, 50)
        RED_UPPER = (10, 255, 255)
        BLACK_LOWER = (0, 0, 0)
        BLACK_UPPER = (180, 255, 30)

        # Track ball movement positions
        ball_positions = []

        # Draw ball movement lines in the frame
        def track_ball_movement(frame, ball_bbox, team_color):
            x1, y1, x2, y2 = ball_bbox
            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            ball_positions.append(ball_center)
            for i in range(1, len(ball_positions)):
                cv2.line(frame, ball_positions[i-1], ball_positions[i], team_color, 2)

        # Helper functions to determine if a detected person belongs to Team 1, Team 2, or is a referee
        def is_team1(frame, x1, y1, x2, y2):
            player_region = frame[int(y1):int(y2), int(x1):int(x2)]
            return detect_color(player_region, WHITE_LOWER, WHITE_UPPER)

        def is_team2(frame, x1, y1, x2, y2):
            player_region = frame[int(y1):int(y2), int(x1):int(x2)]
            return detect_color(player_region, RED_LOWER, RED_UPPER)

        def is_referee(frame, x1, y1, x2, y2):
            person_region = frame[int(y1):int(y2), int(x1):int(x2)]
            return detect_color(person_region, BLACK_LOWER, BLACK_UPPER)

        # Process each frame to detect objects and track ball/player movement
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Run YOLO model for object detection
                with torch.amp.autocast('cuda'):
                    results = model(frame)
                    bboxes = results.xyxy[0].numpy()

                    # Iterate over each detection and process accordingly
                    for bbox in bboxes:
                        x1, y1, x2, y2, conf, cls_id = bbox
                        label = results.names[int(cls_id)]

                        # Track ball and draw its movement
                        if label == 'sports ball':
                            track_ball_movement(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 0, 255))

                        # Track players and color frames based on team or referee
                        elif label == 'person':
                            if is_team1(frame, x1, y1, x2, y2):
                                color = (255, 255, 255)  # White for Team 1
                            elif is_team2(frame, x1, y1, x2, y2):
                                color = (0, 0, 255)      # Red for Team 2
                            elif is_referee(frame, x1, y1, x2, y2):
                                color = (0, 0, 0)        # Black for Referee
                            else:
                                color = (0, 255, 0)      # Green for unassigned

                            # Draw bounding box and label around detected players
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Write processed frame to output video
                out.write(frame)

            except Exception as e:
                return jsonify({'message': f'Error processing frame: {str(e)}'}), 500

        # Release video resources
        cap.release()
        out.release()

        return jsonify({'message': 'Video processed successfully', 'processed_file': filename}), 200
    except Exception as e:
        return jsonify({'message': f'Error processing video: {str(e)}'}), 500

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
