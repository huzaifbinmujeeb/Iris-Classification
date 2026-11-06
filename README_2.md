# Iris Flower Classification

Task 1 of my CodeAlpha Data Science internship. Given four measurements of an Iris flower (sepal length and width, petal length and width), predict which of three species it is: setosa, versicolor or virginica.

Everything is in `CodeAlpha_Iris_Classification.ipynb`. It is already run, so the charts and results show up on GitHub.

## The data

`Iris.csv` is the classic dataset: 150 flowers, 50 of each species, no missing values. I dropped the `Id` column (it is only a row number) and removed the `Iris-` prefix from the species names. Three rows repeat exactly; I kept them since they are separate flowers that just happen to have the same measurements.

## What I did

1. Looked at the data with scatter plots, box plots and a correlation table.
2. Split it 80/20 into train and test, stratified so each species is equally represented in both.
3. Trained three models: logistic regression, K-nearest neighbours (k=5) and a random forest. Logistic regression and KNN get their features scaled first.
4. Checked each one on the 30 test flowers with accuracy, confusion matrices and a classification report. Because 30 flowers is small (one mistake is about 3 points), I also ran 5-fold cross-validation.
5. Looked at which measurements the random forest relies on.

## Results

| Model | Test accuracy (30 flowers) | 5-fold CV accuracy |
|---|---|---|
| K-Nearest Neighbours | 93.3% | 97.3% |
| Random Forest | 90.0% | 96.0% |
| Logistic Regression | 93.3% | 95.3% |

- KNN scores highest in cross-validation, but the differences between the three are small (a single flower on the test set), so I would not call any of them clearly better.
- Setosa was never confused with anything. Every mistake was between versicolor and virginica, which overlap a bit.
- Petal length and petal width carry most of the weight (about 86% of the random forest's feature importance). Sepal width barely matters.

## Limits

Iris is a small, easy dataset with well-separated classes, so these scores say little about how the models would do on a harder problem. I used one split plus cross-validation and did not tune any hyperparameters. Trying different values of k, grid search, and several random seeds would be the next step.

## Run it

```
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook CodeAlpha_Iris_Classification.ipynb
```

Keep `Iris.csv` in the same folder as the notebook.

## Files

- `CodeAlpha_Iris_Classification.ipynb` the analysis
- `Iris.csv` the dataset
