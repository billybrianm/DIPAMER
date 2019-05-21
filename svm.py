import time

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np

start = time.time()


features = np.load('features.npy')
labels = np.load('labels.npy')

print("[INFO] matriz de caracteristicas: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.25, random_state=42)

clf = svm.SVC(gamma=0.001, tol=1e-3)  # Definicao do Kernel linear com gama 0.001 do KNN

# Treinar o modelo utilizando os conjuntos de treinamento
clf.fit(trainFeat, trainLabels)

# Realizar a predicao no conjunto de testes
y_pred = clf.predict(testFeat)


print("Precisao:",metrics.accuracy_score(testLabels, y_pred))

print("Classificacao %s:\n%s\n"
      % (clf, metrics.classification_report(testLabels, y_pred)))

end = time.time()
print(end - start)
