Practical 1:Diabetes dataset for linear regression practical
 

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima- indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)
print(data.head())
print(data.isnull().sum())
data['BMI'].hist(bins=20)
plt.title('BMI Distribution') 
plt.xlabel('BMI') 
plt.ylabel('Frequency') plt.show()
print(data.describe())
X = data.drop('BMI', axis=1) 
y = data['BMI'] 
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}") print(f"Test set size: {X_test.shape[0]}")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred) print(f"Mean Squared Error (MSE): {mse}")
r2 = r2_score(y_test, y_pred) print(f"R-squared (R²): {r2}")
plt.scatter(y_test, y_pred) 
plt.title("Actual vs Predicted BMI") 
plt.xlabel("Actual BMI") 
plt.ylabel("Predicted BMI") 
plt.show()

Practical2: Implement Logistic Regression(iris dataset)


import seaborn as sns 
import pandas as pd 
import numpy as np
df=sns.load_dataset('iris') 
df.head()

df['species'].unique() 

df.isnull().sum() 

df=df[df['species']!='setosa'] 
df.head()

df['species']=df['species'].map({'versicolor':0,'virginica':1}) 
df.head()

X=df.iloc[:,:-1]
y=df.iloc[:,-1] 
X

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
from sklearn.linear_model import LogisticRegression 
classifier=LogisticRegression()

from sklearn.model_selection import GridSearchCV 
parameter={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30, 40,50],'max_iter':[100,200,300]}

classifier_regressor=GridSearchCV(classifier,param_grid=parameter,scorin g='accuracy',cv=5)
classifier_regressor.fit(X_train,y_train)

print(classifier_regressor.best_params_)

print(classifier_regressor.best_score_)

y_pred=classifier_regressor.predict(X_test) \
from sklearn.metrics import accuracy_score,classification_report 
score=accuracy_score(y_pred,y_test)
print(score)

print(classification_report(y_pred,y_test))

sns.pairplot(df,hue='species')    

df.corr()                                                        	
               

Practical3: Implements Multinomial Logistic Regression (Iris Dataset)

import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix from sklearn.decomposition import PCA
data = load_iris()
X = data.data y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(y, data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train_scaled, y_train) y_pred = model.predict(X_test_scaled) print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names)) print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred) print(cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
        plt.title('Confusion Matrix') plt.xlabel('Predicted') plt.ylabel('True') plt.show()
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled) X_test_pca = pca.transform(X_test_scaled)
model_pca = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model_pca.fit(X_train_pca, y_train) plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1, 100),
np.linspace(X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1, 100))
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]) Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis') plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=100)
plt.title("Multinomial Logistic Regression Decision Boundaries (PCA Reduced Data)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2') plt.colorbar(label='Class') plt.show()
 
PRACTICAL 4: Implement SVM Classifier.
import numpy as np
import matplotlib.pyplot as plt from sklearn import datasets
from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
 
# Load the Iris dataset iris = datasets.load_iris()
X = iris.data[:, :2] # We only take the first two features (sepal length and sepal width) y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Standardize the features scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test)
# Create and train the SVM classifier
svm = SVC(kernel='linear', random_state=42) svm.fit(X_train, y_train)
 
# Make predictions
y_pred = svm.predict(X_test)
 
# Evaluate performance
print("Accuracy Score:", accuracy_score(y_test, y_pred)) print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
# Visualization of decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
 
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) Z = Z.reshape(xx.shape)
            Plotting the decision boundary plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
 
# Plot the training points
   plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, marker='o', edgecolors='k', label="Train Data")
   plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='s', edgecolors='k', label="Test Data")
   plt.title('SVM Decision Boundary (Iris Dataset)') plt.xlabel('Sepal Length')
   plt.ylabel('Sepal Width') plt.legend()
    plt.show()

 

 





                                     	
 





import tensorflow as tf import pandas as pd import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data" column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(dataset_url, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].astype(int)
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='') X = dataset.drop('MPG', axis=1)
y = dataset['MPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test) model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape[1],)), Dense(64, activation='relu'),
Dense(1) # Output layer for regression
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae']) history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=2) print(f'\nTest Mean Absolute Error: {mae:.2f} MPG')
