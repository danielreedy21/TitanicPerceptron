import numpy as np
import pandas as pd
import math


# returns 1 if dot product is greater than 0, otherwise returns 0
def predict_one_row(input, weights):
    dot_product = np.dot(input, weights)
    if (dot_product>0):
        return 1
    else:
        return 0

# loops through the inputs and creates/returns a list of weights
def train_model(inputs, Y_actuals, learning_rate, epochs):
    
    # create list of weights that is the same size as one row
    num_attributes = len(inputs[0])
    weights = np.array([0.0]*num_attributes)


    # perform number of epochs
    for epoch in range(epochs):
        print(f'performing epoch number {epoch+1}...')

        # create miss counter
        num_missed = 0

        for idx, input in enumerate(inputs):
            # calculate prediction
            y_predicted = predict_one_row(input, weights)

            # check that y actual matches the prediction. If it does not, then change weights
            if (Y_actuals[idx]-y_predicted == 1):
                weights += learning_rate*input
                num_missed += 1
            elif (Y_actuals[idx]-y_predicted == -1):
                weights -= learning_rate*input
                num_missed += 1
        
        # stop the algorithm when you reach convergence
        if (num_missed==0):
            return weights
        
    return weights


# prints the accuracy of a given model
def test_model_accuracy(inputs, Y_actuals, weights):
    
    correct_count = 0
    
    for idx, input in enumerate(inputs):
        if (Y_actuals[idx] == predict_one_row(input,weights)):
            correct_count += 1
    
    correct_percent = 100*(correct_count/len(inputs))
    print(f'''The weights {weights} have an accuracy of %{correct_percent}''')


# main method
def main():

    # Read the data
    data = pd.read_csv('separable.csv')

    # separate the data by y from the inputs
    y_actuals = np.array(data['obese'])
    inputs = np.array(data.drop(columns=['obese']))
    
    # run the algorithm on the training data
    weights = train_model(inputs,y_actuals,0.1,30)
    print(weights)

    # test the model on the test data
    test_model_accuracy(inputs,y_actuals,weights)

# execution
if __name__=='__main__':
    main()