import numpy as np
import time
import warnings
from collections import deque
from scipy.signal import butter, filtfilt, detrend
from sklearn.decomposition import FastICA

class RPPGMonitor:
    def __init__(self, target_fps=30, window_seconds=10):
        self.target_fps = target_fps
        self.window_size = target_fps * window_seconds
        
        self.buffer = []
        self.times = []
        
        # param per ammortizzare i frame grezzi tra un secondo e l'altro
        self.internal_bpm = None 
        
        # Sliding Window per 8 secondi
        self.bpm_window = deque(maxlen=8)
        self.display_bpm = "Buffering..."
        
        self.last_display_update = time.time()
        self.UPDATE_INTERVAL = 1.0  
        
        self.debug_stats = {}
        
        # configurazione coerente con fastICA.m (deflation, pow3)
        self.ica = FastICA(
            n_components=3, 
            algorithm='deflation', 
            fun='cube', 
            max_iter=1000, 
            tol=0.0001,
            random_state=42
        )

    def add_frame(self, image, landmarks, img_w, img_h):
        try:
            # 1. ROI (fronte)
            pts = [10, 9, 67, 297]
            x_coords = [int(landmarks[i].x * img_w) for i in pts]
            y_coords = [int(landmarks[i].y * img_h) for i in pts]
            
            x_min, x_max = max(0, min(x_coords)), min(img_w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(img_h, max(y_coords))
            
            if x_max <= x_min or y_max <= y_min:
                return self.display_bpm
                
            roi = image[y_min:y_max, x_min:x_max]
            
            # 2. Media spaziale dei canali RGB 
            b_avg, g_avg, r_avg = np.mean(roi, axis=(0, 1))
            
            self.buffer.append([r_avg, g_avg, b_avg])
            self.times.append(time.time())
            
            # 3. pop dal buffer
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)
                self.times.pop(0)
                
        except Exception as e:
            pass

    def estimate_hr(self):
        N_samples = len(self.buffer)
        
        # minimo 5 secondi per far convergere l'ICA
        if N_samples < self.target_fps * 5:
            # calcola i secondi rimanenti (5 - secondi attuali nel buffer)
            remaining_sec = 5 - int(N_samples / self.target_fps)
            self.debug_stats = {"Status": f"Buffering ({N_samples}/{int(self.target_fps*5)})"}
            return f"Buffering... ({remaining_sec}s)"
            
        time_diff = self.times[-1] - self.times[0]
        if time_diff == 0: return self.display_bpm
        
        actual_fps = (N_samples - 1) / time_diff
        
        sig = np.array(self.buffer) # Shape: (window_size, 3)
        
        # Detrend (Rimuove lente variazioni di luce)
        sig = detrend(sig, axis=0)
        
        # Normalizzazione (Media zero, varianza unitaria per FastICA)
        mean = np.mean(sig, axis=0)
        std = np.std(sig, axis=0)
        std[std == 0] = 1 
        normalized_sig = (sig - mean) / std
        
        # 4. Applica Scikit-Learn FastICA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") 
                ic = self.ica.fit_transform(normalized_sig)
        except Exception:
            self.debug_stats = {"Status": "ICA Conv Failed"}
            return self.display_bpm
            
        # 5. Filtro Butterworth (nel range 48-180 BPM)
        lowcut = 0.8  # 48 BPM
        highcut = 3.0 # 180 BPM
        nyq = 0.5 * actual_fps
        
        if lowcut >= nyq or highcut >= nyq:
            return self.display_bpm
            
        b, a = butter(2, [lowcut/nyq, highcut/nyq], btype='band')
        
        max_power = 0
        best_bpm = 0
        best_component = -1
        
        # Zero-Padding
        N_pad = max(2048, len(normalized_sig) * 4)
        
        # 6. componente indipendente con il segnale più forte
        for i in range(ic.shape[1]):
            component = ic[:, i]
            
            # passa-banda
            filtered_component = filtfilt(b, a, component)
            
            # FFT
            fft_mag = np.abs(np.fft.rfft(filtered_component, n=N_pad))
            freqs = np.fft.rfftfreq(N_pad, d=1.0/actual_fps)
            
            # picco
            valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            if len(valid_idx) == 0:
                continue
                
            peak_idx = valid_idx[np.argmax(fft_mag[valid_idx])]
            power = fft_mag[peak_idx]
            
            if power > max_power:
                max_power = power
                best_bpm = freqs[peak_idx] * 60.0
                best_component = i
                
        # 7. Smoothing sui frame grezzi (solo per stabilizzare il picco letto a livello base)
        if best_bpm > 0:
            if self.internal_bpm is None:
                self.internal_bpm = best_bpm
            else:
                self.internal_bpm = 0.8 * self.internal_bpm + 0.2 * best_bpm
                
        # 8. LOGICA DI NORMALIZZAZIONE (ogni 1s)
        current_time = time.time()
        if self.internal_bpm is not None:
            if current_time - self.last_display_update >= self.UPDATE_INTERVAL:
                
                if len(self.bpm_window) > 0:
                    current_avg = sum(self.bpm_window) / len(self.bpm_window)
                    
                    if self.internal_bpm > current_avg * 1.25:
                        pass 
                    else:
                        self.bpm_window.append(self.internal_bpm)
                        new_avg = sum(self.bpm_window) / len(self.bpm_window)
                        self.display_bpm = f"{int(new_avg)}"
                else:
                    self.bpm_window.append(self.internal_bpm)
                    self.display_bpm = f"{int(self.internal_bpm)}"
                    
                self.last_display_update = current_time


        self.debug_stats = {
            "Buffer (s)": round(time_diff, 1),
            "Fs (Hz)": round(actual_fps, 1),
            "Raw HR": round(best_bpm, 1),
            "Max Pwr": round(max_power, 2),
            "Comp": best_component
        }
            
        return self.display_bpm