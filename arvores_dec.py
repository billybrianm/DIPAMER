from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image

features = np.load('features.npy')
labels = np.load('labels.npy')


(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

modelo = tree.DecisionTreeClassifier()

modelo.fit(trainFeat, trainLabels)

y_pred = modelo.predict(testFeat)


dot_data = StringIO()
export_graphviz(modelo, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


dot_data = tree.export_graphviz(modelo, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("dipamer")
