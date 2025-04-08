import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd

# Load dataset
file_path = 'bmi.csv'
labels_df = pd.read_csv(file_path).head(10000)

# Map categories to numeric labels
class_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2}
labels_df['class_label'] = labels_df['Category'].map(class_mapping)

# Paths to image folders
front_images_path = "C:\\Users\\mouni\\OneDrive\\Desktop\\BMI_dataset\\dataset\\front\\front"
side_images_path = "C:\\Users\\mouni\\OneDrive\\Desktop\\BMI_dataset\\dataset\\side\\side"

# Load images
def load_images(image_ids, folder_path, target_size=(128, 128)):
    images = []
    for img_id in image_ids:
        img_path = os.path.join(folder_path, f"{img_id}.jpg")
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img) / 255.0
            images.append(img)
        else:
            images.append(np.zeros((*target_size, 3)))  # Handle missing images
    return np.array(images)

# Load front and side images for the first 10,000 IDs
image_ids = labels_df['id']
front_images = load_images(image_ids, front_images_path)
side_images = load_images(image_ids, side_images_path)

# Split data into train (80%) and test (20%)
X_front_train, X_front_test, X_side_train, X_side_test, y_train, y_test = train_test_split(
    front_images, side_images, labels_df['class_label'], test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

# Define the CNN model
def build_model(input_shape):
    # Front image input branch
    front_input = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(front_input)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Flatten()(x1)

    # Side image input branch
    side_input = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(side_input)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Flatten()(x2)

    # Combine branches
    combined = Concatenate()([x1, x2])
    x = Dense(128, activation='relu')(combined)
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs=[front_input, side_input], outputs=output)
    return model

# Build and compile the model
input_shape = (128, 128, 3)
model = build_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [X_front_train, X_side_train], y_train_onehot,
    validation_data=([X_front_test, X_side_test], y_test_onehot),
    epochs=10,
    batch_size=32
)

# Predictions
y_pred_probs = model.predict([X_front_test, X_side_test])
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Metrics calculation
mae = mean_absolute_error(y_test, y_pred_classes)
mse = mean_squared_error(y_test, y_pred_classes)
r2 = r2_score(y_test, y_pred_classes)
pearson_corr, _ = pearsonr(y_test, y_pred_classes)



# Print metrics with accuracy in percentage
accuracy = history.history['accuracy'][-1] * 100  # Convert training accuracy to percentage
test_accuracy = model.evaluate([X_front_test, X_side_test], y_test_onehot)  # Evaluate on the test set
test_accuracy_percentage = test_accuracy[1] * 100  # Convert test accuracy to percentage

print(f"Training Accuracy: {accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy_percentage:.2f}%")  # The second element of the result is accuracy
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Pearson Correlation Coefficient: {pearson_corr}")

# Function to predict BMI category from an image
def predict_bmi(face_image_path, side_image_path):
    # Load and preprocess the images
    face_img_front = load_img(face_image_path, target_size=(128, 128))
    face_img_front = img_to_array(face_img_front) / 255.0

    face_img_side = load_img(side_image_path, target_size=(128, 128))
    face_img_side = img_to_array(face_img_side) / 255.0

    # Predict the class
    prediction = model.predict([np.expand_dims(face_img_front, axis=0), np.expand_dims(face_img_side, axis=0)])
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map the predicted class back to BMI category
    bmi_category = {0: 'Underweight', 1: 'Normal', 2: 'Overweight'}
    print(f"Predicted BMI category: {bmi_category[predicted_class]}")

# Test the model with your face and your friend's face
your_face_front = "f1.jpg"
your_face_side = "s1.jpg"

friend_face_front = "f2.jpg"
friend_face_side = "s2.jpg"

print("Testing with your face:")
predict_bmi(your_face_front, your_face_side)

print("\nTesting with your friend's face:")
predict_bmi(friend_face_front, friend_face_side)
