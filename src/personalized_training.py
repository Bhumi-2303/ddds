import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

base_model_path = "../models/mobilenetv2_base.h5"
personalized_data_dir = "../personalized_data"
save_path = "../models/personalized_model.h5"

# Load the pre-trained base model
model = load_model(base_model_path)

# Unfreeze last few layers for fine-tuning
for layer in model.layers[-20:]:
    layer.trainable = True

# Prepare the data generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    personalized_data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    personalized_data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
    subset="validation"
)

# Compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=5)

model.save(save_path)
print("✅ Personalized model trained and saved at:", save_path)
