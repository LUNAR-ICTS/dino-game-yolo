import cv2
from ultralytics import YOLO
import os

# Inicializa o modelo YOLO
model = YOLO("yolov8s.pt")

# Lê todos os arquivos da pasta "cats"
path = "cats/"
files = os.listdir(path)

# Cria a pasta "results" caso não exista
os.makedirs("results", exist_ok=True)

# Processa cada imagem na pasta "cats"
for file in files:
    # Lê a imagem
    img = cv2.imread(os.path.join(path, file))
    
    # Processa a imagem com o modelo YOLO
    results = model(img)  # Isso retorna uma lista de Results

    # Percorre os resultados para cada imagem processada
    for result in results:
        # Obtém a imagem com as detecções
        annotated_img = result.plot()  # .plot() retorna a imagem com anotações

        # Exibe a imagem com as detecções
        cv2.imshow("Detecções", annotated_img)
        cv2.waitKey(0)

        # Salva a imagem com as detecções
        cv2.imwrite(os.path.join("results", file), annotated_img)

cv2.destroyAllWindows()
