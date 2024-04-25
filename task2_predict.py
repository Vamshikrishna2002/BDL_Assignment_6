from PIL import Image
import numpy as np

def predict_digit(model,data):
    """
    Predicts the digit from the given data point using the provided model.

    Parameters:
        model (Sequential): The Keras Sequential model.
    Returns:
        str: The predicted digit as a string.
    """ 
    data = np.array(data)
    data=data/max(data)
    print(data)
    print('hello')
    prediction = model.predict(data.reshape(1,-1))
    digit = np.argmax(prediction)
    return str(digit)