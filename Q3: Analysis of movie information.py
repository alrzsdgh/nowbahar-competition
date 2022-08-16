import pandas as pd

df = pd.read_csv('../input/movie-analyse/train.csv')

# I) How many unique movies are there in this dataset?
print('number of unique movies:',len(df['Movie'].unique()))

# II) In this dataset, from what year there are the most movies?
a = df['Year']
a_list = a.tolist()
for i in set(a_list):
    print(f'{i}:{a_list.count(i)}')

# III) In this dataset, what is the highest number of marks given by users?
b = df['Rating']
b_list = b.tolist()
for i in set(b_list):
    print(f'{i}:{b_list.count(i)}')
