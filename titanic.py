import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression #线性回归
from sklearn.cross_validation import KFold #交叉验证库，将测试集进行切分交叉验证取平均
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
pd.options.display.max_columns = 100
import numpy as np
import pylab as plot
params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
}
plot.rcParams.update(params)



# 读取训练机和测试集
titanic_train = pd.read_csv('data/train.csv')
titanic_test = pd.read_csv("data/test.csv")
# 一共有10项属性(survival, pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked)
# 数据预处理
# 对缺失值用平均值填充

"""
# loc定位到哪一行，将titanic['Sex'] == 'male'的样本Sex值改为0
titanic_train.loc[titanic_train['Sex'] == 'male', 'Sex'] = 0
titanic_train.loc[titanic_train['Sex'] == 'female', 'Sex'] = 1
# 用最多的填
titanic_train['Embarked'] = titanic_train['Embarked'].fillna('S')
titanic_train.loc[titanic_train['Embarked'] == 'S', 'Embarked'] = 0
titanic_train.loc[titanic_train['Embarked'] == 'C', 'Embarked'] = 1
titanic_train.loc[titanic_train['Embarked'] == 'Q', 'Embarked'] = 2
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())
# 测试集采取相同的操作
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
"""

# extracting and then removing the targets from the training data
targets = titanic_train.Survived
titanic_train.drop(['Survived'], 1, inplace=True)

# merging train data and test data for future feature engineering
# we'll also remove the PassengerID since this is not an informative feature
combined = titanic_train.append(titanic_test)
combined.reset_index(inplace=True)
combined.drop(['index','PassengerId'], inplace=True, axis=1)

def status(feature):
    print('Processing', feature, ': ok')

# 先打印一下测试集里有哪些Title
""""
titles = set()
for name in titanic_train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)
"""

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

combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
# a map of more aggregated title
# we map each title
combined['Title'] = combined.Title.map(Title_Dictionary)
status('Title')


# 利用Sex，Pclass和Title分组后的中位数来填补缺失的年龄
grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) &
        (grouped_median_train['Title'] == row['Title']) &
        (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]
# a function that fills the missing values of the Age variable
combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
status('age')


# 把Title展开为多个属性 one hot编码
combined.drop('Name', axis=1, inplace=True)
# encoding in dummy variable
titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, titles_dummies], axis=1)
# removing the title variable
combined.drop('Title', axis=1, inplace=True)
status('names')


# 利用训练集的平均值填补Fare
combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
status('fare')


# 利用最多的S来填补缺失
combined.Embarked.fillna('S', inplace=True)
# dummy encoding
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)
status('embarked')


# 利用U来填补缺失,用开头字母来简化Cabin
train_cabin, test_cabin = set(), set()
for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')

for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')
combined.Cabin.fillna('U', inplace=True)
# mapping each Cabin value with the cabin letter
combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
# dummy encoding ...
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
combined = pd.concat([combined, cabin_dummies], axis=1)
combined.drop('Cabin', axis=1, inplace=True)
status('cabin')


# 用1表示Male,用0表述Famale
combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
status('Sex')


# encoding into 3 categories:
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
# adding dummy variable
combined = pd.concat([combined, pclass_dummies], axis=1)
# removing "Pclass"
combined.drop('Pclass', axis=1, inplace=True)
status('Pclass')


# 处理Ticket
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'
combined['Ticket'] = combined['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)
status('Ticket')


# Parch和Sibsp构建一个家族：分为singel，small family 和 large family
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
# introducing other features based on the family size
combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
status('family')
predictors = ['Pclass_1','Pclass_2','Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Title_Master',
              'Title_Miss' , 'Title_Mr'  ,'Title_Mrs',  'Title_Officer','Title_Royalty' ,'Cabin_A' ,
              'Cabin_B','Cabin_C' , 'Cabin_D','Cabin_E' , 'Cabin_F','Cabin_G' , 'Cabin_T',
              'Cabin_U','Embarked_C',  'Embarked_Q', 'Embarked_S','Ticket_LP', 'Ticket_WEP', 'Ticket_Fa',
              'Ticket_A5', 'Ticket_XXX', 'Ticket_SCParis', 'Ticket_SCPARIS', 'Ticket_SCAH', 'Ticket_PP',
               'Ticket_STONOQ', 'Ticket_AQ3', 'Ticket_AS', 'Ticket_WC', 'Ticket_FCC', 'Ticket_SOTONOQ',
                'Ticket_C', 'Ticket_SCA3', 'Ticket_SP', 'Ticket_SCA4', 'Ticket_SOC', 'Ticket_FC', 'Ticket_LINE',
                 'Ticket_SWPP', 'Ticket_SC', 'Ticket_STONO', 'Ticket_PPP', 'Ticket_A4', 'Ticket_SCOW', 'Ticket_SOPP',
                  'Ticket_AQ4', 'Ticket_SOTONO2', 'Ticket_CASOTON', 'Ticket_PC', 'Ticket_CA', 'Ticket_A', 'Ticket_SOP', 'Ticket_STONO2'
]

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

targets = pd.read_csv('data/train.csv', usecols=['Survived'])['Survived'].values
train = combined.iloc[:891]
test = combined.iloc[891:]
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train[predictors], targets)
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train[predictors])
test_reduced = model.transform(test[predictors])
""""
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 10))
plt.show()
"""

# Use grid-search to find the optimized parameters
run_gs =False
if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8,10],
        'n_estimators': [50, 20,10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [3, 5, 10],
        'min_samples_leaf': [1, 3,5,10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                               )

    grid_search.fit(train[predictors], targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'bootstrap': False, 'min_samples_leaf': 5, 'n_estimators': 50,
                  'min_samples_split': 5, 'max_features': 'log2', 'max_depth': 10}
    model = RandomForestClassifier(**parameters)
    model.fit(train[predictors], targets)

"""    
cw = 'balanced'
algorithms=[
#logreg = LogisticRegression()
LogisticRegressionCV(random_state=1),
RandomForestClassifier(min_samples_leaf= 3, n_estimators=50, min_samples_split=10, max_depth=10),
GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
]
logreg_cv=LogisticRegressionCV(random_state=1)
rf=RandomForestClassifier(min_samples_leaf= 3, n_estimators=50, min_samples_split=10, max_depth=10)
gboost=GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
#ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=8, class_weight=cw), n_estimators=50)
full_predictions = []
for alg in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train, targets)
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test.astype(float))[:,1]
    full_predictions.append(predictions)
predictions = (full_predictions[0] + full_predictions[1]*2 + full_predictions[2]) / 4
models = [logreg_cv, rf, gboost]
for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print ('CV score = {0}'.format(score))
    print('****')
rf.fit(train, targets)
"""
predictions = model.predict(test[predictors])
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission3.csv', index=False)

"""
# 用到的特征
# 1.线性回归
# alg = LinearRegression()
# 2.逻辑回归
# alg= LogisticRegression(random_state=1)
# 3.梯度提升树 迭代决策树
# alg= GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
# 4.弱学习算法”提升(boost)为“强学习算法
# alg= AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=3, class_weight=cw), n_estimators=50)
# 5.神经网络(多层感知器分类器)
# alg = MLPClassifier(hidden_layer_sizes=(100, 50, 50), solver='sgd', max_iter=400, learning_rate="adaptive", alpha=0.5, early_stopping=True)
# 6.随机森林
# alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
# 7.朴素贝叶斯(MultinomialNB 假设特征的先验概率为多项式分布 另外还有GaussianNB 假设特征的先验概率为正态分布；MultinomialNB 假设特征的先验概率为多项式分布)
# alg = MultinomialNB(alpha=0.01)
# 8.KNN
# alg = KNeighborsClassifier(n_neighbors=10)
# 9.决策树
# alg = tree.DecisionTreeClassifier()
# 10.支持向量机SVM
# alg = SVC(kernel='rbf', probability=True)
"""

