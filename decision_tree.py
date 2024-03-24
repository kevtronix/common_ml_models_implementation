'''
Decision Tree Implementation
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 


# Sample Data 
X = np.array([[1], [2], [3], [4]])  
y = np.array([0, 1, 0, 1]) 


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model 
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Predictions
predictions = model.predict(x_test)


print(f'Accuracy: {accuracy_score(y_test, predictions)}')