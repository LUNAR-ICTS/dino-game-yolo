import cv2
from ultralytics import YOLO
import time


def process_video(input_video_path, output_video_path, model_path='yolo11x.pt', conf=0.5, device='cuda'):
    # Carregar modelo YOLO
    model = YOLO(model_path)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo.")
    
    # Obter informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar o writer para salvar o vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    prev_time = time.time()
    
    try:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processar frame com YOLO
            results = model.predict(source=frame, conf=conf, device=device)
            annotated_frame = results[0].plot()
            
            # Calcular FPS
            current_time = time.time()
            fps_display = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Exibir FPS no vídeo
            cv2.putText(
                annotated_frame, f"FPS: {fps_display:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            
            # Escrever o frame processado no vídeo de saída
            out.write(annotated_frame)
            
            # Exibir progresso
            frame_number += 1
            print(f"Processando frame {frame_number}/{frame_count}", end='\r')
    
    except Exception as e:
        print(f"Erro durante a execução: {e}")
    
    finally:
        cap.release()
        out.release()
        print("\nProcessamento concluído!")


if __name__ == "__main__":
    input_video = "VideoB.mp4"  # Caminho do vídeo de entrada
    output_video = "output.mp4"  # Caminho do vídeo de saída
    process_video(input_video, output_video)
