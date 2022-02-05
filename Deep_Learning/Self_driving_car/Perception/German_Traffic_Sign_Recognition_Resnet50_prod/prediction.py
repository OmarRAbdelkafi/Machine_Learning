import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def Predict_evaluation(pre_trained_model, X_test, Y_test):
    preds = pre_trained_model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

def Verify_predictions(X_test, Y_test_orig, pre_trained_model, classes):
    prediction = pre_trained_model.predict(X_test)
    pred = np.argmax(prediction, axis=-1)

    plt.figure(figsize=(30,30))
    for i in range(20):
        ax = plt.subplot(5,10,i+1)
        ax.imshow(X_test[i])
        ax.set_title(f'predicted: {classes[pred[i]]}\nactual:{classes[Y_test_orig[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def My_prediction(img_path, input_shape, pre_trained_model, classes):
    img = Image.open(img_path)
    img = img.resize((input_shape[0],input_shape[1]))     # reasizing image
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    print('Input image shape:', x.shape)
    plt.imshow(img)
    plt.show()
    prediction = pre_trained_model.predict(x)
    print("Class prediction = ", prediction)
    print("Class nbr:", np.argmax(prediction))
    print("Class name:", classes[np.argmax(prediction)])
