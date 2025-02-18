import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import deque

def initialize_camera(camera_id=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir a câmera.")
    return cap

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    return fps, current_time

def process_frame(model, frame, conf=0.5, device=0, half=True):
    results = model.predict(source=frame, conf=conf, device=device, half=half)
    return results

def display_fps(frame, fps, position=(10, 30), color=(255, 255, 255), border_color=(0, 0, 0)):
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def box_centers(results):
    centers = []
    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            continue  
        for box in result.boxes:
            if not hasattr(box, 'xyxy') or box.xyxy is None or len(box.xyxy) == 0:
                continue  
            coords = box.xyxy[0].cpu().numpy()  
            if len(coords) < 4:
                continue
            x_min, y_min, x_max, y_max = coords[:4]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            centers.append((int(x_center), int(y_center)))
    return centers

def draw_centers(frame, centers, color=(0, 255, 0), radius=5, thickness=-1):
    last_x, last_y = None, None  
    try:
        for (x, y) in centers:
            cv2.circle(frame, (x, y), radius, color, thickness)
            last_x, last_y = x, y  
        if last_x is None or last_y is None:
            raise ValueError("Nenhuma bounding box detectada")
    except Exception as e:
        print(f"[Aviso] Nenhuma bounding box detectada no frame: {e}")
        last_x, last_y = None, None  
    return frame, last_x, last_y

class JumpDetector:
    def __init__(self, threshold=35, persistence=20, smoothing=15, min_frames=2):
        self.previous_y = None
        self.threshold = threshold
        self.jump_counter = 0
        self.persistence = persistence
        self.smoothing = smoothing
        self.min_frames = min_frames
        self.y_history = deque(maxlen=smoothing)
        self.jump_frame_count = 0

    def detect_jump(self, y_center):
        self.y_history.append(y_center)
        if len(self.y_history) < self.smoothing:
            return False  
        smoothed_y = np.mean(self.y_history)

        if self.previous_y is None:
            self.previous_y = smoothed_y
            return False  

        if self.previous_y - smoothed_y > self.threshold:
            self.jump_frame_count += 1  
        else:
            self.jump_frame_count = 0  

        jump_detected = self.jump_frame_count >= self.min_frames

        if jump_detected:
            self.jump_counter = self.persistence  
            print("🟢 PULO DETECTADO!")

        self.previous_y = smoothed_y
        return jump_detected

    def should_display_jump_text(self):
        if self.jump_counter > 0:
            self.jump_counter -= 1
            return True
        return False

def draw_jump_text(frame, jump_detector):
    if jump_detector.should_display_jump_text():
        text = "PULO!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (frame.shape[1] - 150, 50)  
        color = (0, 0, 255)  
        thickness = 2
        cv2.putText(frame, text, position, font, 1, (0, 0, 0), 4, cv2.LINE_AA)  
        cv2.putText(frame, text, position, font, 1, color, thickness, cv2.LINE_AA)  

# Variável global para ativar/desativar o grid
grid_enabled = False

def draw_grid(frame, spacing=50):
    """
    Desenha um grid horizontal no vídeo.

    :param frame: Frame da câmera.
    :param spacing: Espaçamento entre as linhas do grid.
    :return: Frame com o grid desenhado.
    """
    global grid_enabled
    if not grid_enabled:
        return frame  # Se o grid estiver desativado, retorna o frame original

    height, width, _ = frame.shape
    color = (255, 255, 255)  # Cor branca para as linhas do grid
    thickness = 1  # Espessura das linhas

    # Desenha as linhas horizontais
    for y in range(0, height, spacing):
        cv2.line(frame, (0, y), (width, y), color, thickness)

    return frame

def toggle_grid():
    """
    Alterna o estado do grid entre ativado e desativado.
    """
    global grid_enabled
    grid_enabled = not grid_enabled
    print(f"Grid {'ativado' if grid_enabled else 'desativado'}")

def main():
    model = YOLO('yolo11n.pt')
    cap = initialize_camera(camera_id=0, width=640, height=480)
    prev_time = 0
    jump_detector = JumpDetector(threshold=10, persistence=20)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível capturar o frame da câmera.")
                break

            fps, prev_time = calculate_fps(prev_time)
            results = process_frame(model, frame)
            bounding_box_centers = box_centers(results)
            annotated_frame = results[0].plot()
            annotated_frame, x, y = draw_centers(annotated_frame, bounding_box_centers)
            annotated_frame = draw_grid(annotated_frame)
            
        
            if x is not None and y is not None:
                jump_detected = jump_detector.detect_jump(y)
                if jump_detected:
                    print(f"PULO DETECTADO em Y={y}")
            
            
            draw_jump_text(annotated_frame, jump_detector)
            display_fps(annotated_frame, fps)
            cv2.imshow('Detecção ao Vivo', annotated_frame)

            # Tecla para alternar o grid (G)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                toggle_grid()
            elif key == ord('q'):  # Fecha a aplicação se apertar 'Q'
                break

    except Exception as e:
        print(f"Erro durante a execução: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
