# nowbahar competition

This repository contains codes that address the questions of the NOWBAHAR competition. This is a special [programming competition](https://quera.org/contest/assignments/40246/problems) for the Iranian new year that was held by Amirkabir University, Faculty of Computer Science in March 2022. This competition is a good combination of different parts of programming such as data and machine learning. 


## Question 1: Analysis of students' information

In the provided dataset, there is information about a number of students by which the following questions have to be answered.

I) How many students have been examined in this dataset in general?

II) How many of these students are girls?

III) How many of these students are less than 17 years old and at the same time have income?

IV) How many students have more than 10 absences, yet their study time is the most (4)?


## Question 2: Prediction of students' grades

Following the previous question, we want to predict students' exam scores. Prepare a model that predicts students' grades with the information provided in the dataset.

***I used a Random Forest model that achieves a score of 108 out of 150 on test data***


## Question 3: Analysis of movie information

in the provided dataset there is information about various movies by which the following questions must be answered.

I) How many unique movies are there in this dataset?

II) In this dataset, from what year there are the most movies?

III) In this dataset, what is the highest number of marks given by users?


## Question 4: Movie rating

In continuation of the previous question and by analyzing the dataset, design a model that predicts the user's rating for that movie by seeing new comments.

***I used word embedding method with a simple DNN model that achieves a score of 82 out of 150 on test data***


## Question 5: Food classification

you are required to train a model in order to classify the foods accurately.

***I implemented the transfer learning method for this question. The efficientNetV2B3 model trained on the ImageNet dataset was used as the basic model. The final model achieved a score of 136 out of 150 on test data***

***The dataset of this question can be fined [here](https://www.dropbox.com/s/206jgywtz6s0irn/Q5%20dataset.zip?dl=0)***


## Question 6: number of fingers

We have a database of black and white photos of hands showing the numbers 0 to 5 with their fingers. You are asked to prepare a model that predicts the number shown in the photo by taking similar photos.

***I designed a deep convolutional neural network with 15 layers that achieves a score of 150 out of 150 on test data***

## Question 7: Spam email detection

There is a database of spam and non-spam emails. By checking these emails, design a model that detects new emails as spam or not.

***I used the word embedding method and extract the 1000 most used words as input features. By using a Random Forest model on these features of every email, the final score was 148 out of 150.***
