from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, "data", "train")
test_path = os.path.join(BASE_DIR, "data", "test")
print(train_path)
print(test_path)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)
model = Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7,activation="softmax"))
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15
)
model.save("../models/emotion_model.h5")
print("Model Saved Successfully!")