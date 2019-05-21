import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

inicio = time.time()

rawImages = np.load('rawImages.npy')
features = np.load('features.npy')
labels = np.load('labels.npy')

print("[INFO] matriz de pixels: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] matriz de caracteristicas: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

print("[INFO] avaliando precisao em raw pixel...")
model = KNeighborsClassifier(1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] precisao em raw pixel: {:.2f}%".format(acc * 100))

from sklearn.metrics import classification_report
y_pred = model.predict(testRI)
print(classification_report(testRL,y_pred))

raw = time.time()
print(raw - inicio)

print("[INFO] avaliando precisao em histograma...")
model = KNeighborsClassifier(1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] precisao em histograma: {:.2f}%".format(acc * 100))
y_pred = model.predict(testFeat)
print(classification_report(testLabels,y_pred))

fim = time.time()
print(fim-raw)
print(fim-inicio)