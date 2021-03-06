import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from imutils import paths
import cv2
import numpy as np
from joblib import dump, load
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def get_model():
    # these were identified by running find_model_params
    # params = {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': True}

    # params = {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10,'bootstrap': True}
    # eval_model
    # [0.79245283 0.77358491 0.77358491 0.7254902  0.78431373]
    # 0.7698853126156123

    """
    params = {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 90,'bootstrap': True}
    [0.81203008 0.7593985  0.82706767 0.77692308 0.84615385]
    0.8043146327356853
    """

    params = {'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto',
              'max_depth': None, 'bootstrap': True}
    """
    [0.82485876 0.78285714 0.83428571 0.83333333 0.87356322]
    0.8297796331858283
    """
    # clf = RandomForestClassifier(**params)

    clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')

    # clf = KNeighborsClassifier(n_neighbors=93, p=1, weights="uniform")

    return clf


def find_logreg_model_params(X, y):
    """
    {'solver': 'saga', 'penalty': 'l2', 'C': 0.0001}
    :param X:
    :type X:
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


def find_model_params(X, y):
    model = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(f"Random Parameter Grid:")
    print(random_grid)

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

    operation = "eval_model"  # "random_sample" #"save_model"

    if operation == "model_params":
        X, y, _ = get_image_data()
        find_logreg_model_params(X, y)
    elif operation == "review_predictions":
        model = get_model()
        X, y, images = get_image_data()
        review_incorrect_predictions(model, X, y, images)
    elif operation == "eval_model":
        model = get_model()
        X, y, _ = get_image_data()
        scores = cv_score_model(model, X, y)
        print(scores)
        print(np.mean(scores))
    elif operation == "confusion_matrix":
        model = get_model()
        X, y, images = get_image_data()
        cv_predict_cm_model(model, X, y, images)
    elif operation == "save_model":
        model = get_model()
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
            image = image.flatten()
            actual = imagePath.split("/")[-2]
            pred = model.predict([image])
            print(f"Actual: {actual}, Pred: {directions[pred[0]]}")
