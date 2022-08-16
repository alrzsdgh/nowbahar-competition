# Import needed libraries

import pandas as pd

# import the dataset as a pandas dataframe
# data_path : directory to dataset

data_path = '../input/school1/X_train.csv'
df = pd.read_csv(data_path)

# I) How many students have been examined in this dataset in general?

print('number of unique students:',len(df['StudentID'].unique()))

#__________
# II) How many of these students are girls?

a = df['sex']
counter = 0
for i in a:
    if i == 'F':
        counter = counter + 1
print('number of female students:',counter)
#__________
# III) How many of these students are less than 17 years old and at the same time have income?

b = df[['age', 'paid']]
counter = 0
for i in b.values:
    if i[0]<17 and i[1] == 'yes':
        counter = counter + 1
print('number of students younger than 17 and paid:',counter)
#__________
# IV) How many students have more than 10 absences, yet their study time is the most (4)?

c = df[['absences','studytime']]
counter = 0
for i in c.values:
    if i[0] > 10 and i[1] == 4:
        counter = counter + 1   
print('number of stidents with more than 10 absences and 4 hours of study time:',counter)
