
# coding: utf-8

# In[10]:


# Import required libraries
import numpy as np
import csv
import cv2
import sklearn
import time

from sklearn import preprocessing, metrics, cross_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


# In[11]:


# Constants
data_path = "data/"

csv_params = []
csv_labels = []

# Reading the content of csv file
with open(data_path + 'heart-statlog.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Uncomment this line if file contains headers
    # next(csv_reader, None)
    for each_line in csv_reader:
        csv_params.append(each_line[:-1])
        csv_labels.append(each_line[-1])


# In[13]:


# Converting the data type from list to numpy array
input_params = np.array(csv_params)
input_labels = np.array(csv_labels)


# In[15]:


print("Shape of input params: ", input_params.shape) #(rows, columns)
print("Shape of input labels: ", input_labels.shape) #(rows, columns)


# ##### Headers correspond to each column
# 1. age       
# 2. sex       
# 3. chest pain type  (4 values)       
# 4. resting blood pressure  
# 5. serum cholestoral in mg/dl      
# 6. fasting blood sugar > 120 mg/dl       
# 7. resting electrocardiographic results  (values 0,1,2) 
# 8. maximum heart rate achieved  
# 9. exercise induced angina    
# 10. oldpeak = ST depression induced by exercise relative to rest   
# 11. the slope of the peak exercise ST segment     
# 12. number of major vessels (0-3) colored by flourosopy        
# 13.  thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. input_label -> present/absent

# In[21]:


print("Sample input parameters: ", input_params[0])
print("Corresponding input label: ", input_labels[0])


# In[22]:


# Scaling the input params by using standard scaler
X_scaler = StandardScaler().fit(input_params)
X_scaled = X_scaler.transform(input_params)
print("Scaled Output: ", X_scaled[0])


# In[33]:


# Splitting the data-set in train and test data
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, input_labels, test_size=0.2, random_state=rand_state)


# In[25]:


print("Length of training data: ", len(X_train), len(y_train))
print("Length of test data: ", len(X_test), len(y_test))


# In[34]:


# Training the SVM model
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


# In[36]:


# 57,1,4,140,192,0,0,148,0,0.4,2,0,6,absent
# 67,1,4,160,286,0,2,108,1,1.5,2,3,3,present
validation_array = np.array([57,1,4,140,192,0,0,148,0,0.4,2,0,6])
transformed_input = X_scaler.transform(validation_array.reshape(1, -1))
print(svc.predict(transformed_input))

