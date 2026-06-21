import time
import math
from collections import deque

class DistractionMonitor:
    def __init__(self):
        # intervalli di tempo
        self.LONG_DISTR_THRESHOLD = 5.0
        self.SHORT_DISTR_CUMULATIVE = 10.0
        self.SHORT_DISTR_WINDOW = 30.0
        self.RETURN_TO_NORMAL = 2.0
        self.MICROSLEEP_TH = 4.0
        self.SLEEP_TH = 7.0

        # parametri per inizializzazione timer
        self.owl_start_time = None
        self.lizard_start_time = None
        self.closed_eyes_start_time = None

        # parametri per reset logiche di allarme
        self.focused_start_time = None
        self.eyes_open_start_time = None

        # storico cumulativo su 30s per short distractions
        self.owl_history = deque()
        self.lizard_history = deque()

        # variabili di lock per mantenere l'allarme fino al reset
        self.locked_owl_long = False
        self.locked_owl_short = False
        self.locked_lizard_long = False
        self.locked_lizard_short = False
        self.locked_microsleep = False
        self.locked_sleep = False

        self.last_time = time.time()

    def _distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    # distanza orizzontale per lizard
    def _horizontal_distance(self, p1, p2):
        return abs(p1.x - p2.x)

    def process_frame(self, landmarks):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # estrazione Punti Chiave
        nose = landmarks[4]
        face_left = landmarks[234]
        face_right = landmarks[454]
        
        # occhio sx 
        l_eye_left = landmarks[263]
        l_eye_right = landmarks[362] 
        l_eye_top = landmarks[386]
        l_eye_bot = landmarks[374]
        l_iris = landmarks[473]

        # occhio dx 
        r_eye_left = landmarks[133]  
        r_eye_right = landmarks[33] 
        r_eye_top = landmarks[159]
        r_eye_bot = landmarks[145]
        r_iris = landmarks[468] 

        # 1. EAR 
        ear_left = self._distance(l_eye_top, l_eye_bot) / self._distance(l_eye_left, l_eye_right)
        ear_right = self._distance(r_eye_top, r_eye_bot) / self._distance(r_eye_left, r_eye_right)
        ear = (ear_left + ear_right) / 2.0
        eyes_closed = ear < 0.22 # soglia per occhi chiusi

        # 2. rotazione testa (Owl)
        dist_nose_left = self._distance(nose, face_left)
        dist_nose_right = self._distance(nose, face_right)
        ratio_owl = dist_nose_left / (dist_nose_right + 1e-6)
        is_owl = ratio_owl < 0.6 or ratio_owl > 1.6 # soglia x rotazione, asimmetrico perché rapporto tra pixel

        # 3. movimento occhi (Lizard)
        eye_width_l = self._horizontal_distance(l_eye_left, l_eye_right) + 1e-6
        eye_width_r = self._horizontal_distance(r_eye_left, r_eye_right) + 1e-6
        
        gaze_l = self._horizontal_distance(l_iris, l_eye_left) / eye_width_l
        gaze_r = self._horizontal_distance(r_iris, r_eye_left) / eye_width_r
        gaze_avg = (gaze_l + gaze_r) / 2.0
        
        # se guardo di lato con testa dritta (non OWL) 
        is_lizard = (not is_owl) and (gaze_avg < 0.4 or gaze_avg > 0.6)

        # aggiornamento timer short distractions 
        if is_owl:
            self.owl_history.append((current_time, dt))
        if is_lizard:
            self.lizard_history.append((current_time, dt))
            
        # rimozione contributi più vecchi di 30s
        while self.owl_history and (current_time - self.owl_history[0][0]) > self.SHORT_DISTR_WINDOW:
            self.owl_history.popleft()
        while self.lizard_history and (current_time - self.lizard_history[0][0]) > self.SHORT_DISTR_WINDOW:
            self.lizard_history.popleft()

        # somma dei timer cumulativi
        cumulative_owl = sum([item[1] for item in self.owl_history])
        cumulative_lizard = sum([item[1] for item in self.lizard_history])


        # --- Valutazione Condizioni (Inizi distrazione) ---
        # OWL Continuous
        if is_owl:
            if self.owl_start_time is None: self.owl_start_time = current_time
            if (current_time - self.owl_start_time) >= self.LONG_DISTR_THRESHOLD:
                self.locked_owl_long = True
        else:
            self.owl_start_time = None

        # LIZARD Continuous
        if is_lizard:
            if self.lizard_start_time is None: self.lizard_start_time = current_time
            if (current_time - self.lizard_start_time) >= self.LONG_DISTR_THRESHOLD:
                self.locked_lizard_long = True
        else:
            self.lizard_start_time = None

        # SHORT Distractions trigger
        if cumulative_owl >= self.SHORT_DISTR_CUMULATIVE:
            self.locked_owl_short = True
        if cumulative_lizard >= self.SHORT_DISTR_CUMULATIVE:
            self.locked_lizard_short = True

        # MICROSLEEP / SLEEP
        if eyes_closed:
            if self.closed_eyes_start_time is None: self.closed_eyes_start_time = current_time
            duration = current_time - self.closed_eyes_start_time
            if duration >= self.SLEEP_TH:
                self.locked_sleep = True
            elif duration >= self.MICROSLEEP_TH:
                self.locked_microsleep = True
        else:
            self.closed_eyes_start_time = None

        # --- Valutazione Reset distrazione/sonno ---
        is_focused = not (is_owl or is_lizard)
        if is_focused:
            if self.focused_start_time is None: self.focused_start_time = current_time
            if (current_time - self.focused_start_time) >= self.RETURN_TO_NORMAL:
                self.locked_owl_long = False
                self.locked_owl_short = False
                self.locked_lizard_long = False
                self.locked_lizard_short = False
                self.owl_history.clear()
                self.lizard_history.clear()
                cumulative_owl = 0.0
                cumulative_lizard = 0.0
        else:
            self.focused_start_time = None

        if not eyes_closed:
            if self.eyes_open_start_time is None: self.eyes_open_start_time = current_time
            if (current_time - self.eyes_open_start_time) >= self.RETURN_TO_NORMAL:
                self.locked_microsleep = False
                self.locked_sleep = False
        else:
            self.eyes_open_start_time = None

        # --- Determinazione Stato Finale ---
        warning = False
        driver_state = "Focused"

        if self.locked_sleep:
            driver_state = "Sleep"
            warning = True
        elif self.locked_microsleep:
            driver_state = "Microsleep"
            warning = True
        elif self.locked_owl_long:
            driver_state = "Owl Long distraction"
            warning = True
        elif self.locked_lizard_long:
            driver_state = "Lizard Long distraction"
            warning = True
        elif self.locked_owl_short:
            driver_state = "Owl Short distraction"
            warning = True
        elif self.locked_lizard_short:
            driver_state = "Lizard Short distraction"
            warning = True

        # timer focused
        focus_timer_val = (current_time - self.focused_start_time) if self.focused_start_time is not None else 0.0

        debug_info = {
            "is_owl": is_owl,
            "is_lizard": is_lizard,
            "eyes_closed": eyes_closed,
            "is_focused": is_focused,
            "ratio_owl": round(ratio_owl, 2),
            "gaze_avg": round(gaze_avg, 2),
            "ear": round(ear, 2),
            "cumul_owl (s)": round(cumulative_owl, 1),
            "cumul_lizard (s)": round(cumulative_lizard, 1),
            "focus_timer (s)": round(focus_timer_val, 1)
        }

        return driver_state, warning, debug_info