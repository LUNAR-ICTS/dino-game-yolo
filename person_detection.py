import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
    results = model.predict(source=frame, conf=conf, device=device, half=half)
    return results

def display_fps(frame, fps, position=(10, 30), color=(255, 255, 255), border_color=(0, 0, 0)):
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# Listas para armazenar os dados
y_positions = []
timestamps = []
jump_timestamps = []

def plot_y_positions():
    """
    Plota um gráfico Scatter da posição Y ao longo do tempo, destacando os pulos.
    """
    global timestamps, y_positions, jump_timestamps

    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps, y_positions, color='blue', label="Posição Y")  # Posição normal
    plt.scatter(jump_timestamps, [y_positions[timestamps.index(t)] for t in jump_timestamps],
                color='red', label="Pulo detectado")  # Pulos em vermelho

    plt.xlabel("Tempo (frames)")
    plt.ylabel("Posição Y")
    plt.title("Posição Y ao longo do tempo")
    plt.legend()
    plt.grid(True)
    plt.show()

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

# Lista para armazenar os últimos valores de Y
y_history = deque(maxlen=10)  # Usa os últimos 10 valores para suavizar a detecção
jump_counter = 0  # Controla a exibição de "PULO!"

def detect_jump(y_center, threshold=35, min_frames=2, persistence=20):
    """
    Detecta pulos usando uma média móvel simples.
    
    :param y_center: Coordenada Y atual do centro da bounding box.
    :param threshold: Diferença mínima necessária para considerar um pulo.
    :param min_frames: Número mínimo de frames consecutivos antes de confirmar um pulo.
    :param persistence: Quantidade de frames que "PULO!" ficará visível.
    :return: True se um pulo for detectado, False caso contrário.
    """
    global jump_counter

    # Adiciona a nova posição Y ao histórico
    y_history.append(y_center)

    # Aguarda até ter dados suficientes
    if len(y_history) < y_history.maxlen:
        return False

    # Calcula a média dos últimos valores Y
    smoothed_y = np.mean(y_history)

    # Obtém a posição Y anterior (o primeiro valor da lista)
    previous_y = y_history[0]

    # Detecta pulo se houver uma variação grande para cima
    if previous_y - smoothed_y > threshold:
        jump_counter = persistence  # Mantém "PULO!" visível por alguns frames
        return True

    return False

def should_display_jump_text():
    """
    Controla a exibição da mensagem "PULO!".
    """
    global jump_counter
    if jump_counter > 0:
        jump_counter -= 1
        return True
    return False

def draw_jump_text(frame):
    """
    Desenha "PULO!" no canto superior direito do frame.
    """
    if should_display_jump_text():
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
    
    # Define a resolução desejada com proporção 9:16
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width = min(width, int(height * (9 / 16)))  # Ajusta para manter a proporção
    target_height = min(height, int(width * (16 / 9)))  # Garante que não extrapole
    
    prev_time = 0

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
            annotated_frame = draw_grid(annotated_frame)
            

            if x is not None and y is not None:
                timestamps.append(len(timestamps))  # Cada frame é um ponto no tempo
                y_positions.append(y)

                if detect_jump(y):
                    jump_timestamps.append(len(timestamps) - 1)  # Marca o frame do pulo
                    print(f"PULO DETECTADO em Y={y}")

            draw_jump_text(annotated_frame)
            
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

        # Plota o gráfico da posição Y ao longo do tempo
        plot_y_positions()

if __name__ == "__main__":
    main()
