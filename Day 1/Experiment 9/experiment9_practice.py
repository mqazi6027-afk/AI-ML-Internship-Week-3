

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("      M-Tech AI/ML Internship - Experiment 9")
print(" Deep Learning - Neural Networks & Backpropagation")
print("=" * 65)



iris = load_iris()

X = iris.data
y = iris.target

print("\nDataset Loaded Successfully")
print("Total Samples :", len(X))
print("Features :", X.shape[1])



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = Sequential()

model.add(Dense(16, activation="relu", input_shape=(4,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))



model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



print("\nTraining Model...")

model.fit(
    X_train,
    y_train,
    epochs=50,
    verbose=0
)

print("Training Completed Successfully")


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Accuracy :", round(accuracy * 100, 2), "%")



sample = X_test[:5]

predictions = model.predict(sample, verbose=0)

print("\nPredicted Classes")

for i, value in enumerate(predictions.argmax(axis=1), start=1):
    print(f"Sample {i} : {value}")

print("\nExperiment 9 Completed Successfully")
print("=" * 65)