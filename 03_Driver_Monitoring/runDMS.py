import cv2
import mediapipe as mp
import time
import sys
from dms_core.distraction import DistractionMonitor
from dms_core.rppg import RPPGMonitor

DEBUG_MODE = False
if len(sys.argv) > 1 and sys.argv[1].lower() in ['debug', '--verbose']:
    DEBUG_MODE = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]
NOSE_TIP = [45, 4, 275]

cap = cv2.VideoCapture(0)
dms_monitor = DistractionMonitor()
rppg_monitor = RPPGMonitor()

WINDOW_NAME = 'DMS - Driver Monitoring System'
cv2.namedWindow(WINDOW_NAME)

while cap.isOpened():
    success, image = cap.read()
    if not success or image is None:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape

    driver_state = "--"
    show_warning = False
    debug_info = {}
    heart_rate_bpm = "--"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # monitor distrazioni
            driver_state, show_warning, debug_info = dms_monitor.process_frame(face_landmarks.landmark)
            
            # monitor rPPG
            rppg_monitor.add_frame(image, face_landmarks.landmark, img_w, img_h)
            heart_rate_bpm = rppg_monitor.estimate_hr()

            for idx, lm in enumerate(face_landmarks.landmark):
                cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                if idx in LEFT_EYE or idx in RIGHT_EYE:
                    cv2.circle(image, (cx, cy), radius=1, color=(0, 0, 255), thickness=-1)
                if idx in LEFT_IRIS or idx in RIGHT_IRIS:
                    cv2.circle(image, (cx, cy), radius=1, color=(0, 255, 0), thickness=-1)
                if idx in NOSE_TIP:
                    cv2.circle(image, (cx, cy), radius=2, color=(255, 0, 0), thickness=-1)

    # Debug Info 
    if DEBUG_MODE:
        y_offset = 20

        # distrazioni 
        if debug_info:
            cv2.putText(image, "--- DISTRACTION ---", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_offset += 20
            for key, value in debug_info.items():
                color = (0, 0, 255) if value is True else (255, 255, 255)
                cv2.putText(image, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
                
        # rPPG
        if rppg_monitor.debug_stats:
            y_offset += 10
            cv2.putText(image, "--- rPPG ---", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_offset += 20
            for key, value in rppg_monitor.debug_stats.items():
                cv2.putText(image, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

    if show_warning:
        cv2.putText(image, "WARNING!", (img_w//2-100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    if "Buffering" in str(heart_rate_bpm):
        hr_text = heart_rate_bpm
    else:
        hr_text = f"HR: {heart_rate_bpm} BPM"

    cv2.putText(image, hr_text, (10, img_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if driver_state == "Focused":
        color_state = (0, 255, 0)
    elif "Sleep" in driver_state or "Long" in driver_state:
        color_state = (0, 0, 255)
    else:
        color_state = (0, 255, 255)

    text_size = cv2.getTextSize(f"State: {driver_state}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(image, f"State: {driver_state}", (img_w - text_size[0] - 10, img_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_state, 2)

    cv2.imshow(WINDOW_NAME, image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()