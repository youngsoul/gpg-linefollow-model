import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import cv2
import numpy as np
import pprint
from joblib import dump, load
import random

# 0, 1, 2
directions = ["left", "straight", "right"]

image_path_root = "/Users/patrickryan/Development/python/mygithub/gpg3-linefollower/training_data"


def get_image_data():
    imagePaths = list(paths.list_images(image_path_root))
    direction_vector = []
    images_vector = []
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

    return images_vector, direction_vector


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

    params = {'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}
    """
    [0.82485876 0.78285714 0.83428571 0.83333333 0.87356322]
    0.8297796331858283
    """
    # clf = RandomForestClassifier(**params)


    clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')

    # clf = KNeighborsClassifier(n_neighbors=93, p=1, weights="uniform")

    return clf

def find_logreg_model_params(X,y):
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
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        }
    rf_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid,
                                   n_iter=75, cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    rf_random.fit(X,y)
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
    rf_random.fit(X,y)
    print(rf_random.best_params_)

def cv_eval_model(model, X, y):
    s = time.time()
    scores = cross_val_score(model, X, y, cv=5)
    e = time.time()
    print(f"Cross Value took: {(e-s)} seconds")
    return scores

def train_save_model(model, X, y):
    model.fit(X,y)
    dump(model, "gpg3_line_follower_model.sav")


if __name__ == '__main__':

    operation = "eval_model" # "random_sample" #"save_model"

    if operation == "model_params":
        X, y = get_image_data()
        find_logreg_model_params(X,y)
    elif operation == "eval_model":
        model = get_model()
        X, y = get_image_data()
        scores = cv_eval_model(model, X, y)
        print(scores)
        print(np.mean(scores))
    elif operation == "save_model":
        model = get_model()
        X, y = get_image_data()
        train_save_model(model ,X, y)
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

