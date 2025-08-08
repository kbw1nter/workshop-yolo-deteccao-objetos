##--CONFIGURAR AMBIENTE DE EXECUÇÃO PRA GPU--##

##--- INSTALANDO ULTRALYTICS--- ##
!pip install ultralytics

# --- IMPORTAÇÕES ---
from google.colab import files         # Para fazer upload dos arquivos no Colab
from ultralytics import YOLO            # Biblioteca YOLOv8
import matplotlib.pyplot as plt         # Para mostrar imagens
import matplotlib.image as mpimg        # Para carregar imagens

# --- UPLOAD DAS IMAGENS ---
uploaded = files.upload()               # Janela para selecionar e enviar uma ou mais imagens

# --- MOSTRA QUAIS ARQUIVOS FORAM ENVIADOS ---
print("Arquivos enviados:")
for filename in uploaded.keys():
    print(f"- {filename}")

# --- ESCOLHA DA IMAGEM PELO NOME ---
img_name = input("Digite o nome exato da imagem que deseja processar: ")

# --- VERIFICA SE A IMAGEM ESTÁ ENTRE OS ENVIADOS ---
if img_name not in uploaded:
    print("Erro: imagem não encontrada! Confira o nome e tente novamente.")
else:
    # --- CARREGA O MODELO YOLOv8 ---
    model = YOLO('yolov8n.pt')

    # --- EXECUTA A DETECÇÃO NA IMAGEM ESCOLHIDA ---
    results = model(img_name)

    # --- MOSTRA A IMAGEM ORIGINAL ---
    imagem = mpimg.imread(img_name)
    plt.imshow(imagem)
    plt.title(f"Imagem Original: {img_name}")
    plt.axis('off')
    plt.show()

    # --- MOSTRA A IMAGEM COM AS CAIXAS DE DETECÇÃO ---
    plt.imshow(results[0].plot())
    plt.title(f"Detecções: {img_name}")
    plt.axis('off')
    plt.show()