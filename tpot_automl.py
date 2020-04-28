from train_model import get_image_data
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    _X, _y = get_image_data()

    X = np.asarray(_X)
    y = np.asarray(_y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.2)

    # -*- coding: utf-8 -*-

    """This file is part of the TPOT library.

    TPOT was primarily developed at the University of Pennsylvania by:
        - Randal S. Olson (rso@randalolson.com)
        - Weixuan Fu (weixuanf@upenn.edu)
        - Daniel Angell (dpa34@drexel.edu)
        - and many more generous open source contributors

    TPOT is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    TPOT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

    """

    import numpy as np

    # Check the TPOT documentation for information on the structure of config dicts
    tpot_config = {

        # Classifiers
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },

        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        'sklearn.svm.LinearSVC': {
            'penalty': ["l1", "l2"],
            'loss': ["hinge", "squared_hinge"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        }
    }

    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=3,
                                        random_state=42, verbosity=2, max_time_mins=60,
                                        config_dict=tpot_config)

    pipeline_optimizer.fit(X_train, y_train)
    print(f"Accuracy on test set: {pipeline_optimizer.score(X_test, y_test)}")
    pipeline_optimizer.export('tpot_exported_pipeline.py')
