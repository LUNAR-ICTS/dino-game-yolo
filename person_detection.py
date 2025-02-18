import cv2
from ultralytics import YOLO
import time
import numpy as np

def initialize_camera(camera_id=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir a câmera.")
    return cap

def crop_center(frame, target_width, target_height):
    """
    Corta a imagem do centro para a resolução desejada (proporção 9:16).
    """
    h, w, _ = frame.shape

    # Converte os valores para inteiros
    target_width = int(target_width)
    target_height = int(target_height)

    # Calcula as coordenadas para o crop central
    x_start = int((w - target_width) // 2)
    y_start = int((h - target_height) // 2)

    return frame[y_start:y_start + target_height, x_start:x_start + target_width]

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    return fps, current_time

def process_frame(model, frame, conf=0.5, device=0, half=True):
    '''
    Processa um frame de imagem usando o modelo YOLO.
    '''
    results = model.predict(source=frame, conf=conf, device=device, half=half)
    return results

def display_fps(frame, fps, position=(10, 30), color=(255, 255, 255), border_color=(0, 0, 0)):
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def box_centers(results):
    """
    Extrai as coordenadas do ponto central de cada bounding box detectada.

    :param results: Resultados da predição do modelo YOLO.
    :return: Lista de tuplas (x_center, y_center) das bounding boxes detectadas.
    """
    centers = []

    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            continue  # Pula se não houver bounding boxes detectadas
        
        for box in result.boxes:
            if not hasattr(box, 'xyxy') or box.xyxy is None or len(box.xyxy) == 0:
                continue  # Pula se as coordenadas não forem encontradas
            
            # Move o tensor para a CPU antes de converter para NumPy
            coords = box.xyxy[0].cpu().numpy()  
            
            if len(coords) < 4:
                continue  # Evita erro caso falte algum valor

            x_min, y_min, x_max, y_max = coords[:4]  # Obtém as coordenadas corretamente
            
            # Calcula o ponto central
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            centers.append((int(x_center), int(y_center)))  # Converte para inteiros

    return centers

def draw_centers(frame, centers, color=(0, 255, 0), radius=5, thickness=-1):
    """
    Desenha os pontos centrais das bounding boxes no frame.

    :param frame: Imagem do frame capturado.
    :param centers: Lista de tuplas (x_center, y_center) das bounding boxes.
    :param color: Cor do ponto central (padrão: verde).
    :param radius: Raio do círculo desenhado.
    :param thickness: Espessura do círculo (-1 preenche o círculo).
    :return: Frame com os pontos centrais desenhados e o último centro detectado (ou None).
    """
    last_x, last_y = None, None  # Inicializa variáveis para evitar erro

    try:
        for (x, y) in centers:
            cv2.circle(frame, (x, y), radius, color, thickness)
            last_x, last_y = x, y  # Atualiza com o último centro detectado
        
        if last_x is None or last_y is None:
            raise ValueError("Nenhuma bounding box detectada")

    except Exception as e:
        print(f"[Aviso] Nenhuma bounding box detectada no frame: {e}")
        last_x, last_y = None, None  # Evita erro se nenhuma detecção for feita

    return frame, last_x, last_y

class JumpDetector:
    def __init__(self, threshold=30, persistence=15):
        """
        Inicializa o detector de pulos.

        :param threshold: Valor mínimo de variação no eixo Y para considerar um pulo.
        :param persistence: Quantidade de frames que a mensagem "PULO!" deve ficar visível.
        """
        self.previous_y = None
        self.threshold = threshold
        self.jump_counter = 0
        self.persistence = persistence  # Define quantos frames a mensagem "PULO!" ficará na tela

    def detect_jump(self, y_center):
        """
        Detecta se houve um pulo com base na posição Y do centro da bounding box.

        :param y_center: Coordenada Y atual do centro da bounding box.
        :return: True se um pulo for detectado, False caso contrário.
        """
        if self.previous_y is None:
            self.previous_y = y_center
            return False  # Primeiro frame, sem referência para comparar
        
        jump_detected = self.previous_y - y_center > self.threshold  # Variação Y significativa para cima

        if jump_detected:
            self.jump_counter = self.persistence  # Mantém "PULO!" visível por X frames

        # Atualiza a posição Y anterior
        self.previous_y = y_center

        return jump_detected

    def should_display_jump_text(self):
        """ Verifica se ainda devemos exibir "PULO!" no vídeo """
        if self.jump_counter > 0:
            self.jump_counter -= 1
            return True
        return False


def draw_jump_text(frame, jump_detector):
    """
    Exibe "PULO!" no canto superior direito do frame quando um pulo for detectado.
    """
    if jump_detector.should_display_jump_text():
        text = "PULO!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (frame.shape[1] - 150, 50)  # Canto superior direito
        color = (0, 0, 255)  # Vermelho
        thickness = 2

        cv2.putText(frame, text, position, font, 1, (0, 0, 0), 4, cv2.LINE_AA)  # Borda preta
        cv2.putText(frame, text, position, font, 1, color, thickness, cv2.LINE_AA)  # Texto vermelho



def main():
    model = YOLO('yolo11n.pt')
    cap = initialize_camera(camera_id=0, width=640, height=480)
    prev_time = 0
    # Inicializa o detector de pulos
    jump_detector = JumpDetector(threshold=50, persistence=30)

    # Define a resolução desejada com proporção 9:16
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width = min(width, int(height * (9 / 16)))  # Ajusta para manter a proporção
    target_height = min(height, int(width * (16 / 9)))  # Garante que não extrapole

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível capturar o frame da câmera.")
                break

            # Aplica o crop central (garantindo valores inteiros)
            frame = crop_center(frame, target_width, target_height)

            fps, prev_time = calculate_fps(prev_time)
            results = process_frame(model, frame)
            bounding_box_centers = box_centers(results)
            annotated_frame = results[0].plot()
            annotated_frame, x, y = draw_centers(annotated_frame, bounding_box_centers)

            #print(f"Centro da bounding box: ({x}, {y})")
            
            if x is not None and y is not None:                
                # Detecta pulo e atualiza o estado do JumpDetector
                jump_detector.detect_jump(y)

            # Exibe "PULO!" se um pulo for detectado
            draw_jump_text(annotated_frame, jump_detector)
            
            display_fps(annotated_frame, fps)
            cv2.imshow('Deteccao ao Vivo', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Erro durante a execução: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
