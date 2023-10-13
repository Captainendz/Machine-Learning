#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas 
# Read the CSV file
bankrupt = pandas.read_csv("bankrupt.txt", sep=",")


# In[105]:


import os
print("Current Working Directory:", os.getcwd())


# In[106]:


# Using the shape attribute to find the number of observations and variables
num_observations = bankrupt.shape
variables = bankrupt.columns
print("Number of observations:",num_observations)
#print("variables in the Dataframe:", variables)


# In[107]:


# Calculate descriptive statistics using describe()
statistics = bankrupt.describe()
print("\nDescriptive  Statistics:")
print(statistics)


# In[129]:


# Use value_counts() to count the unique values in the target variable
value_counts = bankrupt[target_variable].value_counts()

#print(value_counts)

# Get the number of companies that went bankrupt (assuming 'Bankrupt' is encoded as 1)
num_bankrupt = value_counts.get(1, 0)  # Get the count of 1's (companies that went bankrupt)

# Calculate the percentage of companies that went bankrupt
percentage_bankrupt = (num_bankrupt / len(bankrupt)) * 100

# Check if the dataset is balanced
is_balanced = value_counts.min() / value_counts.max() > 0.5  # Adjust the threshold as needed

# Print the results
print(f"is_balanced: {is_balanced}")
print(f"Number of companies that went bankrupt: {num_bankrupt}")
print(f"Percentage of companies that went bankrupt: {percentage_bankrupt:.2f}%")
print(f"Is the dataset balanced? {'Yes' if is_balanced else 'No'}")


# In[130]:


import pandas as pd

# Import the train and test datasets
x_train = pd.read_csv("x_train.csv", index_col=0)
x_test = pd.read_csv("x_test.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
y_test = pd.read_csv("y_test.csv", index_col=0)

# Display the number of observations in each dataset
num_observations_train_x = x_train.shape[0]
num_observations_test_x = x_test.shape[0]
num_observations_train_y = y_train.shape[0]
num_observations_test_y = y_test.shape[0]

print("Number of Observations in x Train Data:", num_observations_train_x)
print("Number of Observations in x Test Data:", num_observations_test_x)
print("Number of Observations in y Train Data:", num_observations_train_y)
print("Number of Observations in y Test Data:", num_observations_test_y)

# Check the distribution of classes in the target variable in both datasets
target_variable_train = "Bankrupt"
target_variable_test = "Bankrupt"

bankrupt_counts_train = y_train[target_variable_train].value_counts()
bankrupt_counts_test = y_test[target_variable_test].value_counts()

print("\nDistribution of Classes in Train Data:")
print(bankrupt_counts_train)

print("\nDistribution of Classes in Test Data:")
print(bankrupt_counts_test)


# In[99]:


# Calculate proportions in both datasets
proportions_train = bankrupt_counts_train / num_observations_train
proportions_test = bankrupt_counts_test / num_observations_test

# Check if the distribution is similar
is_distribution_similar = proportions_train.equals(proportions_test)

print("\nIs the Distribution of Classes Similar in Both Data Sets?", is_distribution_similar)


# In[100]:


#Logistic Regression

import statsmodels.api as sm
import statsmodels.formula.api as smf
#import pandas as pd

# Concatenate x_train and y_train
bankrupt_train = pd.concat([x_train, y_train], axis=1)

# Define the logistic regression model
model_formula = "Bankrupt ~ ROAC + ROAA + ROAB + TRA + TAGR + DR + WKTA + CTA + CLA + CFOA + CLCA + NITA"

# Fit logistic regression model
logit_model = smf.glm(formula=model_formula, data=bankrupt_train, family=sm.families.Binomial()).fit()

# Display the summary of the logistic regression model
print(logit_model.summary())


# In[79]:


from sklearn.metrics import confusion_matrix, classification_report

# Get the estimated probabilities
yhat_logreg_probs = logit_model.fittedvalues

# Convert probabilities to binary class labels using a threshold of 0.5
yhat = [1 if x > 0.5 else 0 for x in yhat_logreg_probs]

# Print the confusion matrix
conf_matrix = confusion_matrix(bankrupt_train['Bankrupt'], yhat)
print("Confusion Matrix:")
print(conf_matrix)

# Print the classification report
class_report = classification_report(bankrupt_train['Bankrupt'], yhat, digits=3)
print("\nClassification Report:")
print(class_report)


# In[80]:


from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have already fitted the logistic regression model 'logit_model'
# and you have the train data 'bankrupt_train'

# Get the estimated probabilities
yhat_logreg_probs = logit_model.fittedvalues

# Convert probabilities to binary class labels using a threshold of 0.5
yhat = [1 if x > 0.5 else 0 for x in yhat_logreg_probs]

# Print the confusion matrix
conf_matrix = confusion_matrix(bankrupt_train['Bankrupt'], yhat)
print("Confusion Matrix:")
print(conf_matrix)

# Print the classification report
class_report = classification_report(bankrupt_train['Bankrupt'], yhat, digits=3)
print("\nClassification Report:")
print(class_report)


# In[81]:


# Get the estimated probabilities for the test set
yhat_test_logreg_probs = logit_model.predict(x_test)

# Convert probabilities to binary class labels using a threshold of 0.5
yhat_test = [1 if x > 0.5 else 0 for x in yhat_test_logreg_probs]

# Print the confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, yhat_test)
print("Confusion Matrix for Test Set:")
print(conf_matrix_test)

# Print the classification report for the test set
class_report_test = classification_report(y_test, yhat_test, digits=3)
print("\nClassification Report for Test Set:")
print(class_report_test)


# In[82]:


# K-NEAREST NEIGHBOUR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Create a KNN classifier with K = 1
knn_model = KNeighborsClassifier(n_neighbors=1)

# Train the KNN model on the training set
knn_model.fit(x_train, y_train)

# Make predictions on the test set
yhat_knn = knn_model.predict(x_test)

# Print the confusion matrix for the test set
conf_matrix_knn = confusion_matrix(y_test, yhat_knn)
print("Confusion Matrix for KNN Model:")
print(conf_matrix_knn)

# Print the classification report for the test set
class_report_knn = classification_report(y_test, yhat_knn, digits=3)
print("\nClassification Report for KNN Model:")
print(class_report_knn)



# In[83]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Replace this line if your data loading is different
y_train = np.ravel(y_train)

# Create a KNN classifier with K = 1
knn_model = KNeighborsClassifier(n_neighbors=1)

# Train the KNN model on the training set
knn_model.fit(x_train, y_train)

# Now you can make predictions and evaluate the model


# In[84]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

# Having 'x_train' and 'y_train' as training data
# Having 'x_test' and 'y_test' as your testing data

best_k = None
best_balanced_accuracy = 0

for k in range(1, 21):
    # Create a KNN classifier with the current value of K
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the KNN model on the training set
    knn_model.fit(x_train, np.ravel(y_train))
    
    # Make predictions on the test set
    yhat_knn = knn_model.predict(x_test)
    
    # Calculate balanced accuracy for the current K
    current_balanced_accuracy = balanced_accuracy_score(y_test, yhat_knn)
    
    # Print or store the balanced accuracy for each K (optional)
    print(f'K = {k}, Balanced Accuracy = {current_balanced_accuracy}')
    
    # Check if the current model has a higher balanced accuracy
    if current_balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = current_balanced_accuracy
        best_k = k

# Fit the best model (with the chosen K) on the entire training set
best_knn_model = KNeighborsClassifier(n_neighbors=best_k)
best_knn_model.fit(x_train, np.ravel(y_train))

# Make predictions on the test set with the best model
yhat_best_knn = best_knn_model.predict(x_test)

# Calculate performance indicators, confusion matrix, and classification report
conf_matrix_best_knn = confusion_matrix(y_test, yhat_best_knn)
class_report_best_knn = classification_report(y_test, yhat_best_knn, digits=3)

# Print the results
print(f'\nBest K: {best_k}')
print('Confusion Matrix for Best KNN Model:')
print(conf_matrix_best_knn)
print('\nClassification Report for Best KNN Model:')
print(class_report_best_knn)


# In[85]:


# DISCRIMINANT ANALYSIS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

# Assuming 'x_train' and 'y_train' are your training data
# Assuming 'x_test' and 'y_test' are your testing data

# Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, np.ravel(y_train))
yhat_lda = lda_model.predict(x_test)

# Quadratic Discriminant Analysis (QDA)
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(x_train, np.ravel(y_train))
yhat_qda = qda_model.predict(x_test)

# Evaluate LDA
conf_matrix_lda = confusion_matrix(y_test, yhat_lda)
class_report_lda = classification_report(y_test, yhat_lda, digits=3)

# Evaluate QDA
conf_matrix_qda = confusion_matrix(y_test, yhat_qda)
class_report_qda = classification_report(y_test, yhat_qda, digits=3)

# Print the results
print('Linear Discriminant Analysis (LDA):')
print('Confusion Matrix:')
print(conf_matrix_lda)
print('Classification Report:')
print(class_report_lda)

print('\nQuadratic Discriminant Analysis (QDA):')
print('Confusion Matrix:')
print(conf_matrix_qda)
print('Classification Report:')
print(class_report_qda)

print(lda_model.priors_)
print(lda_model.means_)


# In[86]:


# ROC (Receiver Operating Characteristic) Curve

from sklearn.metrics import roc_curve, auc

# Assuming lda_model is your trained Linear Discriminant Analysis (LDA) model
lda_scores = lda_model.predict_proba(x_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, lda_scores)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Display AUC
print('Area Under the Curve (AUC): {:.2f}'.format(roc_auc))


# In[133]:


from sklearn.metrics import roc_curve, auc

# Having lda_model as our trained Linear Discriminant Analysis (LDA) model
lda_scores = lda_model.predict_proba(x_test)[:, 1]

# Calculate ROC curve
fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_test, lda_scores)

# Calculate AUC
roc_auc_lda = auc(fpr_lda, tpr_lda)
print('AUC for LDA:', roc_auc_lda)

# Having qda_model as our trained Quadratic Discriminant Analysis (QDA) model
qda_scores = qda_model.predict_proba(x_test)[:, 1]

# Calculate ROC curve
fpr_qda, tpr_qda, thresholds_qda = roc_curve(y_test, qda_scores)

# Calculate AUC
roc_auc_qda = auc(fpr_qda, tpr_qda)
print('AUC for QDA:', roc_auc_qda)

# Having logit_model as our trained logistic regression model from statsmodels

# Get the predicted probabilities
logit_probs = logit_model.predict(x_test)

# Calculate ROC curve
fpr_logit, tpr_logit, thresholds_logit = roc_curve(y_test, logit_probs)

# Calculate AUC
roc_auc_logit = auc(fpr_logit, tpr_logit)
print('AUC for Logistic Regression (statsmodels):', roc_auc_logit)

# H knn_model is your trained KNN model with the chosen value of K
knn_scores = knn_model.predict_proba(x_test)[:, 1]

# Calculate ROC curve
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_scores)

# Calculate AUC
roc_auc_knn = auc(fpr_knn, tpr_knn)
print('AUC for KNN:', roc_auc_knn)




# In[134]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Assuming lda_model is your trained Linear Discriminant Analysis (LDA) model
lda_scores = lda_model.predict_proba(x_test)[:, 1]
fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_test, lda_scores)
roc_auc_lda = auc(fpr_lda, tpr_lda)

# Assuming qda_model is your trained Quadratic Discriminant Analysis (QDA) model
qda_scores = qda_model.predict_proba(x_test)[:, 1]
fpr_qda, tpr_qda, thresholds_qda = roc_curve(y_test, qda_scores)
roc_auc_qda = auc(fpr_qda, tpr_qda)

# Assuming logreg_model is your trained logistic regression model from scikit-learn
logreg_model = LogisticRegression()
logreg_model.fit(x_train, y_train)
logreg_scores = logreg_model.predict_proba(x_test)[:, 1]
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, logreg_scores)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

# Assuming knn_model is your trained KNN model with the chosen value of K
knn_scores = knn_model.predict_proba(x_test)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_scores)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curves for all models
plt.figure(figsize=(8, 6))
plt.plot(fpr_lda, tpr_lda, color='darkorange', lw=2, label='ROC curve for LDA (AUC = {:.2f})'.format(roc_auc_lda))
plt.plot(fpr_qda, tpr_qda, color='green', lw=2, label='ROC curve for QDA (AUC = {:.2f})'.format(roc_auc_qda))
plt.plot(fpr_logreg, tpr_logreg, color='blue', lw=2, label='ROC curve for Logistic Regression (AUC = {:.2f})'.format(roc_auc_logreg))
plt.plot(fpr_knn, tpr_knn, color='purple', lw=2, label='ROC curve for KNN (AUC = {:.2f})'.format(roc_auc_knn))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiple Models')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# In[ ]:




