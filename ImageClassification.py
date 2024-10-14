import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Function to analyze and classify an image
def analyze_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for the model

    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode the predictions into readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 predictions
    return decoded_predictions

# Main function
if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    results = analyze_image(image_path)

    print("\nPredicted categories and probabilities:")
    for result in results:
        print(f"{result[1]}: {result[2]*100:.2f}%")
