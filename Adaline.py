import numpy as np
import pandas as pd
from numpy.random import seed
import random

# IMPLEMENTATION FROM TEXTBOOK
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        # for idx, weight in enumerate(self.w_):
        #     self.w_[idx]= random.random()*2 - 1
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression, we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, 0)



# MY CODE #
def transform_data(data: pd.DataFrame):
    
    # set non-attributse and columns with mostly missing values to 0
    data["Name"] = 0
    data["PassengerId"] = 0
    data["Cabin"] = 0
    
    # transform Pclass
    data["Pclass"] = data["Pclass"]/3

    # transform sex
    data["Sex"] = np.where(data["Sex"] == "male", 0, 1)

    # transform Age and normalize
    data["Age"].mask(pd.isna(data["Age"]), 30.0, inplace=True)
    data["Age"] = data["Age"]/80
    
    # transform Embarked
    data.loc[data["Embarked"] == "S", "Embarked"] = 0.33
    data.loc[data["Embarked"] == "Q", "Embarked"] = 0.66
    data.loc[data["Embarked"] == "C", "Embarked"] = 0.99
    data["Embarked"].mask(pd.isna(data["Embarked"]), 0, inplace=True)

    # transform null fare values into 0 and standardize
    data["Fare"].mask(pd.isna(data["Fare"]), 0, inplace=True)
    data["Fare"] = data["Fare"]/512.3292
    
    # I can't figure out the logic for transforming ticket numbers, so I will set to zero temporarily
    data["Ticket"] = 0

    # remove Survived columns
    data = data.drop(columns=['Survived'])

    # set data to a numpy array
    data_as_numpy = data.to_numpy()
    return data_as_numpy

def display_accuracy(y_pred, y, TestOrTrain: str):
    num_correct_predictions = (y_pred == y).sum()
    accuracy = round((num_correct_predictions / y.shape[0]) * 100,2)

    print(f'''
    Correctly classified {TestOrTrain}ing samples: {num_correct_predictions} out of {y.shape[0]}
    The model has an accuracy of {accuracy}%
    ''')



def main():
    # read the input data
    train_data = pd.read_csv('train.csv')
    train_answers = train_data['Survived']

    test_data = pd.read_csv('test.csv')
    test_answers = pd.read_csv('gender_submission.csv')
    test_data = pd.concat([test_data,test_answers['Survived']], axis=1)


    # transform the input data into numpy array of numbers
    train_data = transform_data(train_data).astype(np.float64)
    test_data = transform_data(test_data)
    # turn the target values into 1d numpy arrays
    train_answers = train_answers.to_numpy().astype(np.float64)
    test_answers = test_answers['Survived'].to_numpy()


    # train the model on training data
    # adal = AdalineGD(n_iter=1000, eta=0.0002853)
    adal = AdalineGD(n_iter=1000, eta=0.00028)
    adal.fit(train_data,train_answers)
    print(f'''
    Weights from fitting the model on the training data:
    {adal.w_}
    ''')
    print(adal.cost_)


    # see accuracy on the training data
    y_pred = adal.predict(train_data)
    display_accuracy(y_pred, train_answers, "train")
    
    # see accuracy on the testing data
    y_pred = adal.predict(test_data.astype(np.float64))
    display_accuracy(y_pred, test_answers, "test")




    # CREATE AND PREDICT WITH RANDOM WEIGHTS
    for idx, weight in enumerate(adal.w_):
        adal.w_[idx]= random.random()*2 - 1
    print(f'''
    Weights from randomization:
    {adal.w_}
    ''')

    y_pred = adal.predict(test_data.astype(np.float64))
    display_accuracy(y_pred, test_answers, "test")



if __name__=='__main__':
    main()