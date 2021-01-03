# Imbalanced Learning

## Introduction

Imbalance is a very common issue in classification problems. It is inherent to some areas like anomaly detection, diagnosis, spam detection or insurance claims among others. The issue in the imbalanced case is that most classification algorithms are doing maximization of the accuracy or a similar measure and if the ratio of the data is 99:1, it will very certainly only predict the majority class and shows an accuracy of 99%, whereas the algorithm is not doing anything at all. First, it is necessary to use a different metric adapted to this kind of problem like the f1-score or the AUC or the area below the ROC (Receiving Operator Characteristic) curve.
The f1-score is:

<img src="Img/f1-score.png" alt="f1-score" width="400">

with

<img src="Img/precision.png" alt="precision" width="400">

and

<img src="Img/recall.png" alt="recall" width="400">

![AUC](Img/ROC.png)

The AUC is the area below the colored curves.

There are some algorithms that naturally deal well with imbalanced datasets like methods based on trees. It is also possible to use cost-sensitive algorithms to put more weights on the minority class for example.

A different approach is to use resampling methods that aim to artificially get a balanced dataset. There are several ways to do that, it is possible to remove observations from the majority class (undersampling), to replicate or to create observations of the minority class (oversampling) or to do a combination of both.

## Datasets chosen

### Credit Card
We have a dataset about credit card frauds with the data of 284797 individuals. There are only 482 frauds in this dataset, so the ration is close to 600:1. It is a binary problem where we have to find abnormal variables.
Insurance claims
The goal with this dataset is to predict whether a client would be interested in vehicle insurance. It is a binary classification problem. We have information about demographics (gender, age, region code type), vehicles (Vehicle Age, Damage) and policy (Premium, sourcing channel) etc. The ratio is close to 5:1.



|interested | not interested |
----------- | -------------- |
| 62531 | 319594 |


### Wine quality

The third dataset is about wine quality. We have to do a multi-classification task as we have 6 possible classes distributed as below.

|Wine Quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------------ |-- |-- |---|---|---|---|---|
| Number of observations | 30 | 216 | 2138 | 2836 | 1079 | 193 |5 |

## Methods to deal with imbalance dataset

The simplest way to do resampling is to do random resampling but in case of undersampling, it leads to a loss of information which can be even more annoying than the imbalance itself. 

### Tomek Links

However there exists more sophisticated methods to lower the defaults of this method. A famous undersampling method is Tomek Links.
It consists in creating pairs of points with each pair containing a point from the minority class and a point from the majority class. Then, we simply remove the points of the majority class on some pairs to reach the ratio we want. It gives a cleaner separation between the two classes and hopefully better results.

![TomekLink](Img/TL_1.png) ![TomekLink](Img/TL_2.png)

Illustration of the undersampling process in the Tomek Links algorithms

### Cluster Centroids

Another way to do undersampling is to compute the centroids of the points in the majority class 
and to replace these points by their centroids. It is a way to lose less information since we are 
‘synthesizing’ the information contained in the majority class. 

![Cluster Centroids](Img/Cluster_Centroids.png)

Illustration of the undersampling process in the Cluster Centroids algorithm

### Cost-sensitive SVM

The Support Vector Machine algorithm looks for an hyperplane which leads to the best separation between the classes. Without adjustments SVM are not very effective in the imbalanced case, but when the errors in the majority class are weighted more heavily we can get very satisfying results.
The SVM minimizes the risk of the hinge loss

<img src="Img/loss_fcn.png" alt="loss function" width="300">

In the cost-sensitive case, it leads to the following cost-sensitive loss function.

![Loss function cost-sensitive](Img/loss_complicated.png)

We can adjust the value of C(i). With the function above, the bigger C(1), the bigger the weight for misclassified 1 and  the bigger C(-1), the bigger the weight for misclassified -1.

![SVMs](Img/SVM.png)

Illustration of the weighted SVM versus non weighted SVM

The decision trees or random forest algorithms often perform well on an imbalanced dataset due to the splitting rule that can force both classes to be addressed. But it is also possible to use cost-sensitive decision trees to reach better results.

## Results obtained with each method on each dataset

## Conclusion

## Bibliography

1. https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/under-sampling/plot_illustration_tomek_links.html#sphx-glr-auto-examples-under-sampling-plot-illustration-tomek-links-py

2. https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

3. https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/under-sampling/plot_comparison_under_sampling.html#sphx-glr-auto-examples-under-sampling-plot-comparison-under-sampling-py

3. https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/
https://arxiv.org/pdf/1212.0975.pdf

4. Masnadi-Shirazi, Vasconcelos and Iranmehr, Cost-sensitive Support Vector Machines, Journal of Machine Learning Research (2015)

5. Krawczyk, B., Woźniak, M., Schaefer, G.: Cost-sensitive decision tree ensembles for effective imbalanced classification. Appl. Soft Comput. 14, 554–562 (2014)
