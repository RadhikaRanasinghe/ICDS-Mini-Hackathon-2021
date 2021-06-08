# ICDS-Mini-Hackathon-2021

This serves the first round competition for ICDS2021 Mini-Hackathon.

In this competition you will work with a challenging dataset consisting of applications received to a nursery. Predict the class of the application status as to whether the applicant would be selected or not, by using the app_status as the dependent variable. The rest of the variables would be independent variables.

Upload the predicted outcome of the test set according to the format provided, to obtain the accuracy of the prediction.

## Technical details

* python 3.8 venv

## Dataset

There are 10 variables in the dataset. The variable app_status can be considered as the class label or the response variable.

#### File descriptions

* train_data.csv - the training set
  
* test_data.csv - the test set
  
* sampleSubmission.csv - a sample submission file in the correct format

#### Data Fields  
* **ID** - unique ID given to an applicant parents - occupation of parents (usual, pretentious, great_pret) 
  
* **has_nurs** - child's nursery (proper, less_proper, improper, critical, very_crit)
  
* **form** - form of the family (complete, completed, incomplete, foster) 
  
* **children** - (children number (1, 2, 3, more) 
  
* **housing** - housing conditions (convenient, less_conv, critical) 
  
* **finance** - financial standing of the family (convenient, inconv) 
  
* **social** - social conditions (non-prob, slightly_prob, problematic) 
  
* **health** - health conditions (recommended, priority, not_recom) 
  
* **app_status** - label/target/response variable (1-selected, 0-not selected)

#### Acknowledgement
This dataset is a modified version of the **UCI Nursery dataset**.

## Rules of the competition 

* Only One Kaggle account to be created per team as mentioned in the email. (You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.)

* Submission limits - Only a maximum of 5 submissions per day are allowed for the competition (You can try five test data submissions per day during the time the competition window is open)

* No private sharing of the response variable of the test dataset allowed. This would lead for disqualification.

* Team mergers are not allowed in this competition.

* Competition timeline

Start date – Monday, 7th of June 2021 at 8.00 a.m  
End date – Sunday, 13th of June 2021 at 11.59 p.m
