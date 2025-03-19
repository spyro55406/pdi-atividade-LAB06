import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def adicionar_texto_na_imagem(imagem, texto, posicao, tamanho_fonte=30, cor=(0, 255, 255)):
    imagem_pil = Image.fromarray(imagem)
    desenhador = ImageDraw.Draw(imagem_pil)
    fonte = ImageFont.truetype("arial.ttf", tamanho_fonte)
    desenhador.text(posicao, texto, font=fonte, fill=cor)
    return np.array(imagem_pil)

def calcular_area(contorno):
    return cv2.contourArea(contorno)

def encontrar_contornos(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(imagem_cinza, 50, 150)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def processar_video(caminho_video):
    captura = cv2.VideoCapture(caminho_video)
    ultrapassagem_detectada = False
    impacto_detectado = False
    contato_anterior = False

    while captura.isOpened():
        sucesso, quadro = captura.read()
        if not sucesso:
            break

        quadro = cv2.resize(quadro, (quadro.shape[1] // 2, quadro.shape[0] // 2))
        resultado = quadro.copy()
        contornos = encontrar_contornos(quadro)

        formas = [(c, calcular_area(c)) for c in contornos]

        if len(formas) >= 2:
            maior_contorno = max(formas, key=lambda x: x[1])[0]
            menor_contorno = min(formas, key=lambda x: x[1])[0]
            cv2.drawContours(resultado, [maior_contorno], -1, (0, 0, 255), 2)

            colidiu = (cv2.boundingRect(menor_contorno)[0] + cv2.boundingRect(menor_contorno)[2] > cv2.boundingRect(maior_contorno)[0] and
                       cv2.boundingRect(menor_contorno)[0] < cv2.boundingRect(maior_contorno)[0] + cv2.boundingRect(maior_contorno)[2] and
                       cv2.boundingRect(menor_contorno)[1] + cv2.boundingRect(menor_contorno)[3] > cv2.boundingRect(maior_contorno)[1] and
                       cv2.boundingRect(menor_contorno)[1] < cv2.boundingRect(maior_contorno)[1] + cv2.boundingRect(maior_contorno)[3])

            if colidiu:
                impacto_detectado = True
                contato_anterior = True
                resultado = adicionar_texto_na_imagem(resultado, "COLISÃO DETECTADA", (500, 50), tamanho_fonte=40, cor=(0, 255, 255))

            elif impacto_detectado and not colidiu:
                if cv2.boundingRect(maior_contorno)[0] < cv2.boundingRect(menor_contorno)[0]:
                    ultrapassagem_detectada = True
                    impacto_detectado = False
                    contato_anterior = False

            if ultrapassagem_detectada and not contato_anterior:
                resultado = adicionar_texto_na_imagem(resultado, "PASSOU A BARREIRA", (500, 50), tamanho_fonte=40, cor=(0, 0, 255))

        cv2.imshow('Saída', resultado)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

caminho_arquivo_video = 'q1B.mp4'
processar_video(caminho_arquivo_video)
