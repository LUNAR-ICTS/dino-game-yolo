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

def get_bounding_box_centers(results):
    """
    Extrai as coordenadas do ponto central de cada bounding box detectada.

    :param results: Resultados da predição do modelo YOLO.
    :return: Lista de tuplas (x_center, y_center) das bounding boxes detectadas.
    """
    centers = []

    for result in results:
        boxes = result.boxes  # Obtém as bounding boxes detectadas
        
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Coordenadas da caixa (canto superior esquerdo e inferior direito)
            
            # Calcula o ponto central
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            centers.append((int(x_center), int(y_center)))  # Converte para inteiros

    return centers

def draw_bounding_boxes(frame, results, color=(0, 255, 0), thickness=2, font_scale=0.5):
    """
    Desenha as bounding boxes detectadas no frame.

    :param frame: Frame da câmera onde as boxes serão desenhadas.
    :param results: Resultados da predição do modelo YOLO.
    :param color: Cor da bounding box (padrão: verde).
    :param thickness: Espessura da linha da bounding box.
    :param font_scale: Tamanho do texto da classe.
    :return: Frame com as bounding boxes desenhadas.
    """
    for result in results:
        boxes = result.boxes  # Obtém as bounding boxes detectadas
        names = result.names  # Obtém os nomes das classes

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Coordenadas inteiras da bounding box
            class_id = int(box.cls[0])  # ID da classe detectada
            confidence = float(box.conf[0])  # Confiança da detecção
            label = f"{names[class_id]}: {confidence:.2f}"  # Nome da classe + confiança
            
            # Desenhar a bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

            # Posição do texto (acima da bounding box)
            text_x, text_y = x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10

            # Fundo do texto (para melhorar a visibilidade)
            cv2.rectangle(frame, (x_min, text_y - 5), (x_min + len(label) * 10, text_y + 5), (0, 0, 0), -1)

            # Texto da classe + confiança
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def main():
    model = YOLO('yolo11n.pt')
    cap = initialize_camera(camera_id=0, width=640, height=480)
    prev_time = 0

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
            #results = process_frame(model, frame)
            annotated_frame = draw_bounding_boxes(frame, results)
            bounding_box_centers = get_bounding_box_centers(results)
            annotated_frame = results[0].plot()

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
