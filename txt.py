import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the model
yolo = YOLO('yolo11n.pt')

PATH = 0

videoCap = cv2.VideoCapture(PATH)

# Verificar se o vídeo abriu corretamente
if not videoCap.isOpened():
    print("Erro ao abrir o vídeo. Verifique o caminho do arquivo.")
    exit()

# Configuração do círculo
radius = 5
color = (255, 0, 0)  # Azul em BGR
thickness = -1

# Vetores para salvar xmed e os frames correspondentes
vetor_x = []
vetor_y = []
frames = []

# Função para obter cores das classes
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
            (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Processamento do vídeo
while True:
    ret, frame = videoCap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura do frame.")
        break

    results = yolo(frame)  # Melhor usar detect() em vez de track()

    for result in results:
        classes_names = result.names  # Obter nomes das classes

        for box in result.boxes:
            if box.conf[0] > 0.4:  # Filtrando por confiança
                cls = int(box.cls[0])

                # Filtrar apenas a classe "pessoa" (classe 0 no YOLOv8)
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertendo coordenadas para int
                    class_name = classes_names[cls]
                    colour = getColours(cls)

                    # Desenhar retângulo e legenda
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, colour, 2)

                    # Calcular centro da bounding box
                    xmed = (x1 + x2) // 2
                    ymed = (y1 + y2) // 2

                    # Desenhar o círculo no meio da bounding box
                    cv2.circle(frame, (xmed, ymed), radius, color, thickness)

                    # Salvar ponto central a cada X frames
                    intervalo = 15  # A cada 20 frames
                    frame_atual = int(videoCap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_atual % intervalo == 0:
                        vetor_y.append(ymed)
                        vetor_x.append(xmed)
                        frames.append(frame_atual)  # Armazena o número do frame correspondente

    # Mostrar o frame
    cv2.imshow('frame', frame)

    # Parar a execução se 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar captura de vídeo
videoCap.release()
cv2.destroyAllWindows()

# Plotar deslocamento de Xmed e Ymed ao longo do tempo
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # Dois subplots na vertical

# Gráfico do deslocamento de Xmed
axs[0].plot(frames, vetor_x, marker='o', linestyle='-', color='b')
axs[0].set_ylim(300, 800)
axs[0].set_xlabel("Número do Frame")
axs[0].set_ylabel("Posição Xmed")
axs[0].set_title("Deslocamento de Xmed ao longo do tempo")
axs[0].grid()

# Gráfico do deslocamento de Ymed
axs[1].plot(frames, vetor_y, marker='o', linestyle='-', color='r')
axs[1].set_ylim(300, 800)  # Ajuste conforme necessário
axs[1].set_xlabel("Número do Frame")
axs[1].set_ylabel("Posição Ymed")
axs[1].set_title("Deslocamento de Ymed ao longo do tempo")
axs[1].grid()

plt.tight_layout()  # Ajusta layout para evitar sobreposição de textos
plt.show()