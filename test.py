import cv2
from ultralytics import YOLO
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Constantes
MILLI = 1000
MAN_HEIGHT_M = 1.7
EARTH_GRAVITY = 9.81
ACCELERATION_ERROR = 0.5 * EARTH_GRAVITY
INTERPOLATION_SPAN = 100
MAX_MILLISECONDS_BETWEEN_JUMPS = 800
MIN_N_FRAMES = 4

# Classe JumpCounter (adaptada)
class JumpCounter:
    def __init__(self):
        self._timestamps = []
        self._boxes = []
        self._all_timestamps = []
        self._all_boxes = []
        self._count = 0
        self._last_jump_timestamp = None

    def _is_height_change(self, new_box):
        if not self._boxes:
            return False
        last_height = self._boxes[-1][3]
        new_height = new_box[3]
        height_change = abs(new_height - last_height) / last_height
        return height_change > 0.1  # 10% de mudança

    def _check_for_jump(self):
        if len(self._boxes) < MIN_N_FRAMES:
            return False

        df = pd.DataFrame({'box': self._boxes}, index=self._timestamps)
        m_to_p_ratio = MAN_HEIGHT_M / df.box.iloc[0][3]
        df['y'] = df.box.apply(lambda r: -r[1] * m_to_p_ratio)

        df.index = pd.to_datetime(df.index, unit='ms')
        interpolated = df.y.resample('1L').interpolate()
        smoothed = interpolated.ewm(span=0.5 * INTERPOLATION_SPAN).mean()

        velocity = (smoothed.diff() * MILLI).ewm(span=INTERPOLATION_SPAN).mean()
        acceleration = (velocity.diff() * MILLI).ewm(span=INTERPOLATION_SPAN).mean()

        df = pd.DataFrame({
            'y': smoothed,
            'v': velocity,
            'a': acceleration.shift(-20)
        })
        df['freefall'] = ((df.a + EARTH_GRAVITY).abs() < ACCELERATION_ERROR)
        df['local_maximum'] = ((df.y.shift(1) < df.y) & (df.y.shift(-1) <= df.y))
        df['high_enough'] = (df.y - df.y.min()) > MAN_HEIGHT_M * 0.1

        if any(df.freefall & df.local_maximum & df.high_enough):
            self._boxes = self._boxes[-MIN_N_FRAMES:]
            self._timestamps = self._timestamps[-MIN_N_FRAMES:]
            return True

        return False

    def count_jumps(self, box, timestamp):
        if box is None:
            return self._count

        if self._is_height_change(box):
            self._timestamps = []
            self._boxes = []
            self._all_timestamps = []
            self._all_boxes = []
            self._count = 0
            self._last_jump_timestamp = None

        self._boxes.append(box)
        self._timestamps.append(timestamp)
        self._all_boxes.append(box)
        self._all_timestamps.append(timestamp)

        if len(self._boxes) < MIN_N_FRAMES:
            return self._count

        if len(self._boxes) > 4 * INTERPOLATION_SPAN:
            self._boxes = self._boxes[:INTERPOLATION_SPAN]
            self._timestamps = self._timestamps[:INTERPOLATION_SPAN]

        if self._check_for_jump():
            if self._last_jump_timestamp and timestamp - self._last_jump_timestamp > MAX_MILLISECONDS_BETWEEN_JUMPS:
                self._count = 0
            self._count += 1
            self._last_jump_timestamp = timestamp

        return self._count

# Funções auxiliares
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

def draw_centers(frame, centers, color=(0, 255, 0), radius=5, thickness=-1):
    for (x, y) in centers:
        cv2.circle(frame, (x, y), radius, color, thickness)
    return frame

def display_fps(frame, fps, position=(10, 30), color=(255, 255, 255), border_color=(0, 0, 0)):
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# Função principal
def main():
    model = YOLO('yolov8n.pt')  # Carrega o modelo YOLO
    cap = initialize_camera(camera_id=1, width=640, height=480)
    prev_time = 0

    # Dicionário para rastrear pessoas e seus JumpCounters
    trackers = defaultdict(JumpCounter)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível capturar o frame da câmera.")
                break

            fps, prev_time = calculate_fps(prev_time)

            # Processa o frame com YOLO
            results = process_frame(model, frame)
            annotated_frame = results[0].plot()

            # Itera sobre as detecções
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # Classe 0 é "pessoa" no YOLO
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                        person_id = int(box.id) if box.id else None

                        if person_id is not None:
                            # Atualiza o JumpCounter para esta pessoa
                            jump_count = trackers[person_id].count_jumps(
                                box=(x_min, y_min, x_max, y_max),
                                timestamp=time.time() * MILLI
                            )

                            # Exibe o contador de pulos no frame
                            cv2.putText(
                                annotated_frame,
                                f"Pulos: {jump_count}",
                                (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 0),
                                2
                            )

            # Exibe o frame
            display_fps(annotated_frame, fps)
            cv2.imshow('Detecção ao Vivo', annotated_frame)

            # Sai do loop se 'q' for pressionado
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Erro durante a execução: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()