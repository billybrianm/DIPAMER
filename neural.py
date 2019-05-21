import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier

features = np.load('features.npy')
labels = np.load('labels.npy')

print("[INFO] matriz de caracteristicas: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

test = time.time()# Geração da Rede Neural
modelo = MLPClassifier(solver='adam', #utilização do classificador adam
                    learning_rate='constant', #taxa de aprendizagem constante e a mesma para todos os neurônios
                    learning_rate_init=1e-3, #aprendizagem inicial de 0.001
                    alpha=1e-5, # valor de regularização L2
                    hidden_layer_sizes=(5, 2), # definição das camadas escondidas
                    max_iter=200, # qtde de epochs
                    random_state=1)

# Treinamento do modelo
modelo.fit(trainFeat, trainLabels)

# Previsao utilizando o dataset de testes
y_pred = modelo.predict(testFeat)


print("Precisao:", metrics.accuracy_score(testLabels, y_pred))

print("Classificacao %s:\n%s\n"
      % (modelo, metrics.classification_report(testLabels, y_pred)))


print(modelo.score(trainFeat, trainLabels))