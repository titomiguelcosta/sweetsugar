# Models

## Linear

Supervised learning. Classification problems.

### Perceptron

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train, y_train)

y_predict = ppn.predict(X_test)

### Logistic regression

Important to ensure that all features are on comparable scales

```
from sklearn.linear_model import LogisticRegression

\# Parameter C is for tweaking regularization, increasing its values means decreasing the regularization strength (increases bias, so if model overfits, we should increase C)

lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train, y_train)
lr.predict_proba([[x1, x2, x3, ...]])
```

### Support vector machines

```
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)
```

If dataset is not linearly separable, you can use the kernel rbf. 

Increasing gamma value, will result in overfitting.

```
from sklearn.svm import SVC

svm = SVC(kernel='rbf', gamma=0.1, C=10.0, random_state=0)
svm.fit(X_train, y_train)
```

### Decision trees

Feature scalling is not necessary. 

Important to set the maxium depth. To high it will overfit.

```
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
```

One advantage is that we can export the result tree.

```
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='filename.dot', feature_names=['x1', 'x2'])
```

### Random forest

It will create a certain number of trees picking random features and make the decision by majority vote.

```
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
```

### K-nearest neighbors

It is an example of a nonparametric model, since it instead of learning weights, it memorizes the training set.

It finds the k closest neighbors, and decides by majority, the label for a sample.

No training set is needed, we use all the samples for building the model. 

Storaging and lookup times are the main issues, as the data increases.

```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X, y)
```

p=2 and metric='minkowski', it will use Euclidean distance.

# Evaluate

```
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)
```