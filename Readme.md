# GoPiGo Line Follower Model Optimizer project

This repo is a support repo for the GoPiGo line following project.

This project was used to explore different models and model hyperparameters before trying to train a model on the Raspberry PI.

Please see my [GitHub Repo](https://github.com/youngsoul/gpg-linefollower) for the project.  This repo has a detailed Readme about the project and process.

## TPOT

This project will use TPOT AutoML to determine the best Model pipeline for the given dataset.

https://epistasislab.github.io/tpot/installing/

`pip install deap update_checker tqdm stopit`

`pip install tpot`


## Models

### LogisticRegression


```text
clf = LogisticRegression(penalty="l2", C=0.1, solver='newton-cg')


Cross Value took: 36.17222595214844 seconds
[0.77435897 0.8        0.77948718 0.76804124 0.81958763]
0.7882950039651071
```


```text
clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga')

Cross Value took: 74.64749574661255 seconds
[0.82564103 0.8        0.8        0.82989691 0.85051546]
0.8212106793550094

```

After cleaning up the samples and throwing out incorrectly labeled images:

```text
clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')

Cross Value took: 67.93020796775818 seconds
[0.84831461 0.85310734 0.87570621 0.8700565  0.89830508]
0.8690979495969022

```

After reviewing samples again:
```text
Cross Value took: 68.83101201057434 seconds
[0.95530726 0.94972067 0.96648045 0.97206704 0.95530726]
0.9597765363128492

```


More training data
```text
clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')

Cross Value took: 88.6061041355133 seconds
[0.96943231 0.9650655  0.9650655  0.97816594 0.96069869]
0.9676855895196507

```

### KNN

TPOT found the following:

`clf = KNeighborsClassifier(n_neighbors=93, p=1, weights="uniform")`

```text
Average CV score on the training set was: 0.8174339174339175
Accuracy on test set: 0.8307692307692308
```

```text
Cross Value took: 11.049602031707764 seconds
[0.82564103 0.77948718 0.81025641 0.81958763 0.84536082]
0.8160666137985725
```
