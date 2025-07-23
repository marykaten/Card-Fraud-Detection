import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

credit_card_data = pd.read_csv('creditcard.csv')

# separating the data of legit and fraudulent transactions for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Under_Sampling Method
# Randomly sample the 492 points in the fraudulent data section
legit_sample = legit.sample(n=492)

# Combine the sample of legit and fraud data
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Uniformly distributed data of legit and fraud
new_dataset['Class'].value_counts()

# Splitting data
X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']

# Testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

# Model Training
model = make_pipeline(StandardScaler(), LogisticRegression())

# Training the Logistic Regression Model with training data
model.fit(X_train, Y_train)

# Initialize StandardScaler
scaler = StandardScaler()

# Accuracy Score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(training_data_accuracy)

#Accuracy Score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(test_data_accuracy)