#Solução para abrir no navegador padrão do usuário
import numpy as np
import webbrowser
import pyautogui
import time
import cv2
import os


# Obtém o caminho absoluto do arquivo HTML
caminho_html = os.path.abspath("dino_game\index.html")

# Abre o arquivo no navegador padrão
webbrowser.open("file://" + caminho_html)

# Espera 2 segundos antes de pressionar a tecla
time.sleep(2)

# Pressiona a tecla 'f11'
pyautogui.press('f11')

time.sleep(1)

i = 0
while i < 3:
    pyautogui.press('ctrl'+'+')
    time.sleep(1)
    i += 1

# Função para verificar se a imagem está na tela
def imagem_na_tela(imagem_path, tempo_limite):
    start_time = time.time()
    while True:
        # Tirar uma captura de tela
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        # Carregar a imagem que estamos procurando
        imagem_procurada = cv2.imread(imagem_path)
        
        # Usar a função matchTemplate do OpenCV para encontrar a imagem
        resultado = cv2.matchTemplate(screenshot, imagem_procurada, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)
        
        # Se a similaridade for maior que 0.9, consideramos que a imagem foi encontrada
        if max_val > 0.9:
            print("Imagem encontrada!")
            if time.time() - start_time > tempo_limite:
                print(f"A imagem ficou visível por {tempo_limite} segundos!")
                break
        else:
            # Caso a imagem não seja encontrada, resetamos o tempo
            start_time = time.time()

        # Pausa de 1 segundo antes de tentar novamente
        time.sleep(1)

# Exemplo de uso
imagem_path = "caminho_para_a_imagem.png"  # Caminho da imagem que você quer detectar
tempo_limite = 5  # Limite de tempo em segundos

imagem_na_tela(imagem_path, tempo_limite)