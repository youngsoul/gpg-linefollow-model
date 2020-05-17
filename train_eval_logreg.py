import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from imutils import paths
import cv2
import numpy as np
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import string

# 0, 1, 2
directions = ["left", "straight", "right"]

image_path_root = "/Users/patrickryan/Development/python/mygithub/gpg3-linefollower/training_data"


def get_image_data():
    imagePaths = list(paths.list_images(image_path_root))
    direction_vector = []
    images_vector = []
    image_names = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        # images should be gray scale but make sure
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"Shape: {image.shape}, Image: {imagePath}")
        flatten_image = image.flatten()
        print(f"Flat Shape: {flatten_image.shape}")
        direction = directions.index(imagePath.split("/")[-2])
        images_vector.append(flatten_image)
        direction_vector.append(direction)
        image_names.append(imagePath)

    return images_vector, direction_vector, image_names


def get_logreg_model():
    """
    This function returns the 'best' or model under test instance
    :return:
    :rtype:
    """
    clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')
    return clf


def find_best_logreg_model_params(X, y):
    """
    {'solver': 'saga', 'penalty': 'l2', 'C': 0.0001}
    :param X: ndarray of shape ( 60*192 ) = 11520
    :type X: ndarray
    :param y:
    :type y:
    :return:
    :rtype:
    """
    model = LogisticRegression()
    random_grid = {
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'penalty': ["l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1],
        'class_weight': [None, 'balanced']
    }
    rf_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid,
                                   n_iter=75, cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    rf_random.fit(X, y)
    print(rf_random.best_params_)

def cv_score_model(model, X, y):
    s = time.time()
    scores = cross_val_score(model, X, y, cv=5)
    e = time.time()
    print(f"Cross Value took: {(e - s)} seconds")
    return scores

def review_incorrect_predictions(model, X, y, image_names):
    preds = cross_val_predict(model, X, y, cv=5)
    directions = ['left', 'straight', 'right']

    for i, pred in enumerate(preds):
        if y[i] != pred and y[i]==0 and pred==2:
            print(f"Actual: {directions[y[i]]}, Pred: {directions[pred]}, File: {image_names[i]}")
            img = cv2.imread(image_names[i])
            plt.imshow(img)
            plt.show()
            cmd = input("1-Use Actual, 2-Create 10 Copies, 5-Use Predicted, 0-delete: ")
            print(cmd)
            cmd = int(cmd.strip())
            if cmd == 1:
                continue
            elif cmd == 2:
                for _ in range(0,10):
                    new_filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
                    new_filepath = image_names[i].replace(image_names[i].split("/")[-1], f"{new_filename}_{directions[y[i]]}.jpg")
                    print(f"Creating new copy: {new_filepath}")
                    cv2.imwrite(new_filepath, img)
            elif cmd == 5:
                new_filename = image_names[i].replace(directions[y[i]], directions[pred])
                print(f"Moving: {image_names[i]} to {new_filename}")
                Path(image_names[i]).replace(new_filename)
            elif cmd == 0:
                Path(image_names[i]).unlink(missing_ok=True)


def cv_predict_cm_model(model, X, y, image_names=None):
    preds = cross_val_predict(model, X, y, cv=5)
    cm = confusion_matrix(y, preds)
    df_cm = pd.DataFrame(
        cm, index=['left', 'straight', 'right'], columns=['left', 'straight', 'right']
    )
    print(df_cm)
    for i, pred in enumerate(preds):
        if y[i] != pred:
            print(f"Actual: {directions[y[i]]}, Pred: {directions[pred]}, File: {image_names[i]}")


def train_save_model(model, X, y):
    model.fit(X, y)
    dump(model, "gpg3_line_follower_model.sav")


if __name__ == '__main__':
    """
    operation:
    set this string to the operation of interest. 
    
    find_best_logreg_model_params: call a function to performa and RandomizedSearchCV for a LogisticRegression model
                                    Once you have model parameters, update the 'get_logreg_model' function to return
                                    a LogisticRegression model with this parameters
    
    review_predictions: Using the best LogisticRegression Model, predict on the training dataset and display the
                        training images that were predicted incorrectly.  This can help you reclassify the image
                        if necessary.
                        
    eval_model: Use the best LogisticRegression Model,  run a cross_val_score on the training
    
    confusion_matrix: Use the best LogisticRegression Model, run a cross_val_score and display the confusion matrix.
                      and list the incorrect predictions
                      
    save_model: Use the best LogisticRegression Model, train and save the model.  Keep in mind you CANNOT transfer 
                this saved model to the GoPiGo.  It will not be able to be loaded and executed.
    
    random_sample: Use the saved model and randomly select 10 images to display with the actual and predicted value.
    """
    operation = "confusion_matrix"  # "random_sample" #"save_model"

    if operation == "find_best_logreg_model_params":
        X, y, _ = get_image_data()
        find_best_logreg_model_params(X, y)
    elif operation == "review_predictions":
        model = get_logreg_model()
        X, y, images = get_image_data()
        review_incorrect_predictions(model, X, y, images)
    elif operation == "eval_model":
        model = get_logreg_model()
        X, y, _ = get_image_data()
        scores = cv_score_model(model, X, y)
        print(scores)
        print(np.mean(scores))
    elif operation == "confusion_matrix":
        model = get_logreg_model()
        X, y, images = get_image_data()
        cv_predict_cm_model(model, X, y, images)
    elif operation == "save_model":
        model = get_logreg_model()
        X, y, _ = get_image_data()
        train_save_model(model, X, y)
    elif operation == "random_sample":
        model = load("gpg3_line_follower_model.sav")
        imagePaths = list(paths.list_images(image_path_root))
        random.shuffle(imagePaths)
        imagePaths = imagePaths[:10]
        for imagePath in imagePaths:
            print(f"Load Image: {imagePath}")
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_flat = image.flatten()
            actual = imagePath.split("/")[-2]
            pred = model.predict([image_flat])
            print(f"Actual: {actual}, Pred: {directions[pred[0]]}")
            cv2.imshow("Line Image", image)
            cv2.waitKey(0)

