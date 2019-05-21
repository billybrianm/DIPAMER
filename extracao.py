from imutils import paths
import numpy as np
import argparse
import cv2
import os
import time

start = time.time()

def img_para_raw(imagem, size=(128, 128)):
    # reescala a imagem para um tamanho fixo e gera uma lista de
    # intensidades de raw pixel em um array
    return cv2.resize(imagem, size).flatten()


def extrair_histograma(image, bins=(8, 8, 8)):
    # extrai um histograma de cores de 3 dimensões do espaço HSV
    # utilizando a quantidade de bins fornecida
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # realiza a normalização da imagem
    cv2.normalize(hist, hist)  # alargamento de contraste

    return hist.flatten()

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", required=True,
	help="caminho para o dataset")
args.add_argument("-k", "--neighbors", type=int, default=1,
	help="# de vizinhos mais proximos para classificacao")
args.add_argument("-j", "--jobs", type=int, default=-1,
	help="# de jobs por distancia k-NN (-1 usa todos os cores disponiveis)")
args = vars(args.parse_args())
print("[INFO] descrevendo imagens...")
imagePaths = list(paths.list_images(args["dataset"]))

rawImages = []
features = []
labels = []
for (i, imagePath) in enumerate(imagePaths):
    # carrega a imagem e extrai seu rótulo de classe para o array label
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # utiliza os algoritmos nas imagens
    pixels = img_para_raw(image)
    hist = extrair_histograma(image)

    # atualiza as matrizes com os valores
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)



    # mostra uma atualização a cada 1000 imagens
    if i > 0 and i % 1000 == 0:
        print("[INFO] processados {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] matriz de pixels: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] matriz de caracteristicas: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

np.save('features.npy',features)
np.save('labels.npy',labels)
np.save('rawImages.npy',rawImages)

end = time.time() - start
print(end)