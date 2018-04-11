# Kaggle-Titanic
Recently, I started to do some small projects on Kaggle. This is my first project on Kaggle alone. There is a summary about this project.

The data are more than 10 dimensions, including Age, Sex, Fare etc. In the project, I first pre-process data. combine the train data and test data, then find more features.
Some pre-processing as follows:
1. Use names to extract the titles and then simply classify some categories, including:
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

2. Group by using the information Sex, Pclass and Title to fill the empty.
3. Use pandas function to get dummies (one hot encoding)
There are more methods to pre-process data, please view source code.

After pre-processing data, i began to use many classification algorithms to classify and predict. Before formally starting using model, need to find the correlation between features (now we have more than 50 features)
Use random forest to select the features, relevant codes as follows:

#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

#clf = clf.fit(train[predictors], targets)

#model = SelectFromModel(clf, prefit=True)

In the model, parameters are very important, here one method called grid-search could help you find optimized parameter for model.

#grid_search.fit(train[predictors], targets)

#model = grid_search

#parameters = grid_search.best_params_

Sometimes, we could use multiple models to vote for the final prediction. However, for the simple data set, Blending different models dose not have good result apparently.
To summarize, this is my first time to try do projects on Kaggle alone. Titanic is the simple project and lots of resources and tutorials could been found online.

Reference: https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
