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

'''
# Função para verificar se a imagem está na tela e pressionar uma tecla
def imagem_na_tela_e_acionar_tecla(tempo_limite, tecla):
    start_time = time.time()

    # Inicializa a variável para armazenar a imagem capturada a cada 10 segundos
    imagem_procurada = None

    while True:
        # Captura uma imagem a cada 10 segundos
        time.sleep(10)
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Se for a primeira captura, define como a imagem inicial
        if imagem_procurada is None:
            imagem_procurada = screenshot
            print("Imagem inicial capturada. Aguardando a detecção...")

        # Usar a função matchTemplate do OpenCV para comparar a imagem
        resultado = cv2.matchTemplate(screenshot, imagem_procurada, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)
        
        # Se a similaridade for maior que 0.9, consideramos que a imagem foi encontrada
        if max_val == 1:
            print("Imagem encontrada!")
            if time.time() - start_time > tempo_limite:
                print(f"A imagem ficou visível por {tempo_limite} segundos!")
                # Ação: Pressionar a tecla
                pyautogui.press(tecla)
                print(f"Tecla '{tecla}' pressionada!")
                break
        else:
            # Caso a imagem não seja encontrada, resetamos o tempo
            start_time = time.time()

# Exemplo de uso
tempo_limite = 10  # Limite de tempo em segundos
tecla = 'f11'   # A tecla que será pressionada

imagem_na_tela_e_acionar_tecla(tempo_limite, tecla)
'''
