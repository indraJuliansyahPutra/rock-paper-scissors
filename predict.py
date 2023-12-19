import numpy as np
import tensorflow as tf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def load_and_predict(image_path, model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    im_array = tf.keras.preprocessing.image.img_to_array(image)
    im_array = im_array / 255.0
    im_input = tf.reshape(im_array, shape=[1, 150, 150, 3])

    # Make prediction
    predict_array = model.predict(im_input)[0]

    # Create a DataFrame for the prediction results
    df = pd.DataFrame(predict_array, columns=['Probability'])
    df['Product'] = ['Paper', 'Rock', 'Scissors']
    df = df[['Product', 'Probability']]

    # Get the predicted label
    predict_label = np.argmax(predict_array)

    # Map the label to the corresponding product
    products = ['Paper', 'Rock', 'Scissors']
    predict_product = products[predict_label]

    return predict_product, df

# Example usage
image_path = 'Stock_Gambar_Tangan_38-3209746817.jpg'  # Replace with the path to your image
model_path = 'result/best_model.h5'  # Replace with the path to your saved model
prediction, probability_df = load_and_predict(image_path, model_path)

print("Predicted Product:", prediction)
print("\nPrediction Probabilities:")
print(probability_df)
