import numpy as np
import pandas as pd
import math
import sys

# prompt the user to decide which data set to use
data_name = input('''Please enter \'train\' to run the algorithm with training data, 
or type \'test\' to run the algorithm with test data:
''')

# read the correct data set, if test data append the 'Survived' column to the data set
if(data_name=='train'):
    data = pd.read_csv('train.csv')
elif(data_name=='test'):
    data = pd.read_csv('test.csv')
    test_answers = pd.read_csv('gender_submission.csv')
    data = pd.concat([data,test_answers['Survived']], axis=1)
else:
    print('Incorrect input, please enter only \'train\' or \'test\'')
    quit()


# create weights for the attributes
pclass_weight = -0.2 # negative because the lower the better chance of survival
sex_weight = 0.3
age_weight = -0.3 # negative because the lower the age the better chances of survival
fare_weight = 0.2

# count variable to track the algorithm's success rate
correct_count = 0
passenger_count = len(data.index)

# Loop through each passenger
for pass_index in range(0,len(data.index)):
   
    survived = data.loc[pass_index,"Survived"]
    survival_score = 0
    survival_guess = 0


    # do necessary data transformations (I am giving every attribute a numeric value between 0 and 1)
        #pclass transformation
    pclass = data.loc[pass_index, "Pclass"]
    pclass_bound_by_1 = pclass*0.33
        
        #sex transformation
    sex = data.loc[pass_index,"Sex"]
    if(sex=='male'):
        sex_bound_by_1 = 0
    elif(sex=='female'):
        sex_bound_by_1 = 1
        
        #age transformation
    age = data.loc[pass_index,"Age"]
    if(math.isnan(age)):
        age_bound_by_1 = .30 # if data is missing, set to average age which is 30
    else:
        age_bound_by_1 = age*.01
        
        #fare transformation
    fare = data.loc[pass_index, "Fare"]
    fare_bound_by_1 = round(fare*0.00195, 5) #0.00195 because that is 1/513 and 513 is the highest fare



    # calculate survival score for each passenger(row)
    survival_score = pclass_weight*pclass_bound_by_1 + sex_weight*sex_bound_by_1 + age_weight*age_bound_by_1 + fare_weight*fare_bound_by_1
    if (survival_score>0):
        survival_guess = 1
    
    # check survival scores against actual survival data
    if(survived==survival_guess):
            correct_count += 1

    # calculate algorithm success rate
    success_percent = round((correct_count/passenger_count)*100, 2)


print(f'''
The fate of this many passengers was guessed correctly: {correct_count}
Out of {passenger_count} passengers
Therefore, the algorithm had a success rate of %{success_percent}
''')