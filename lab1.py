from sklearn.datasets import load_iris
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split


iris_sklearn = load_iris()
iris_df = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)

iris_df['class'] = iris_sklearn.target

dataset_shape = iris_df.shape
num_features = iris_df.shape[1] - 1
target_classes_count = iris_df['class'].value_counts()
missing_percentage = iris_df.isnull().mean() * 100
description = iris_df.describe()

print(dataset_shape, '\n', f'Num of features {num_features}\n', target_classes_count,
      f' Missing percentage: {missing_percentage}\n', description)


correlation_matrix = iris_df.corr()
correlation_by_class = iris_df.groupby('class').corr()


sns.pairplot(iris_df, hue='class', markers=["o", "s", "D"])
plt.show()

plt.imshow(correlation_by_class, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title("Matrix Visualization of DataFrame")
plt.xticks(ticks=np.arange(len(correlation_by_class.columns)), labels=correlation_by_class.columns, rotation=90)
plt.yticks(ticks=np.arange(len(correlation_by_class.index)), labels=correlation_by_class.index)
plt.show()

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

selected_labels = ['sepal_length', 'petal_length']

df_binary = iris_df[iris_df['class'].isin([0, 1])]
X = df_binary.drop(columns=['class'])
y = df_binary['class']


feature_names = X.columns


plt.figure(figsize=(15, 10))
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, (fst, snd) in enumerate(pairs):
    plt.subplot(2, 3, i + 1)
    X_pair = X.iloc[:, [fst, snd]]
    lda_pair = LinearDiscriminantAnalysis()
    lda_pair.fit(X_pair.values, y)

    plt.scatter(X_pair.iloc[:, 0], X_pair.iloc[:, 1], c=y, cmap='coolwarm', edgecolor='k')

    x_min, x_max = X_pair.iloc[:, 0].min() - 1, X_pair.iloc[:, 0].max() + 1
    y_min, y_max = X_pair.iloc[:, 1].min() - 1, X_pair.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = lda_pair.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    plt.xlabel(feature_names[fst])
    plt.ylabel(feature_names[snd])
    plt.title(f'{feature_names[fst]} vs {feature_names[snd]}')


plt.tight_layout()
plt.show()


X = X[['sepal_length', 'sepal_width']]

lda = LinearDiscriminantAnalysis()
lda.fit(X.values, y)

X_reg = X[['sepal_length']].values
y_reg = X['sepal_width'].values
reg = LinearRegression()
reg.fit(X_reg, y_reg)

x_min, x_max = X['sepal_length'].min() - 1, X['sepal_length'].max() + 1
y_min, y_max = X['sepal_width'].min() - 1, X['sepal_width'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))


Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

reg_line_x = np.linspace(x_min, x_max, 100).reshape(-1, 1)
reg_line_y = reg.predict(reg_line_x)

plt.figure(figsize=(12, 12))

plt.plot(reg_line_x, reg_line_y, color='red', linewidth=2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X['sepal_length'], X['sepal_width'], c=y, edgecolor='k', s=20)
plt.xlabel('Длина чашелистика (см)')
plt.ylabel('Ширина чашелистика (см)')
plt.title('Решающая функция LDA')

plt.show()


def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', s=20)
    ax.set_xlabel('Длина чашелистика (см)')
    ax.set_ylabel('Ширина чашелистика (см)')
    ax.set_title(title)


lda = LinearDiscriminantAnalysis()
lda.fit(X.values, y)

svm = SVC(kernel='linear')
svm.fit(X.values, y)

log_reg = LogisticRegression()
log_reg.fit(X.values, y)

naive_bayes = GaussianNB()
naive_bayes.fit(X.values, y)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))


plot_decision_boundary(lda, X, y, axs[0, 0], title='LDA')
plot_decision_boundary(svm, X, y, axs[0, 1], title='SVM')
plot_decision_boundary(log_reg, X, y, axs[1, 0], title='Логистическая регрессия')
plot_decision_boundary(naive_bayes, X, y, axs[1, 1], title='Наивный Байесовский классификатор')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB()
}


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nМатрица ошибок для {model.__class__.__name__}:\n{cm}")

    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


for model_name, model in models.items():
    evaluate_model(model, X_train, y_train, X_test, y_test)
