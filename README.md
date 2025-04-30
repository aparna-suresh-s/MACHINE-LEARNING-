import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

df = pd.read_csv("C:\\Users\\user\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
x = df[['Partner', 'Dependents']].values
y = df.iloc[:, 4].values

le_x = LabelEncoder()
x_encoded = np.copy(x)
x_encoded[:, 0] = le_x.fit_transform(x_encoded[:, 0])
x_encoded[:, 1] = le_x.fit_transform(x_encoded[:, 1])


le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=0)
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'yellow')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color=ListedColormap(('purple', 'yellow'))(i), label=le_y.inverse_transform([j])[0])
plt.title('Telecommunications (Train Set)')
plt.xlabel('Partner (Encoded)')
plt.ylabel('Dependents (Encoded)')
plt.legend()
plt.show()


x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'yellow')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color=ListedColormap(('purple', 'yellow'))(i), label=le_y.inverse_transform([j])[0])
plt.title('Telecommunications (Test Set)')
plt.xlabel('Partner')
plt.ylabel('Dependents')
plt.legend()
plt.show()

