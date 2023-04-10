#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[128]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import  load_model
from tensorflow.keras.utils import plot_model
import random


# In[3]:


df=pd.read_csv("data.csv")


# In[4]:


df.head()


# In[5]:


df['diagnosis'].value_counts()


# ## Cleaning Data

# In[6]:


df.drop('id',axis=1,inplace=True)


# In[7]:


df.drop('Unnamed: 32',axis=1,inplace=True)


# In[ ]:





# ### Transforming categorical classes into numerical data

# In[8]:


df.head()


# In[9]:


print(df.dtypes)


# In[10]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})


# In[11]:


df.head()


# In[12]:


from sklearn import preprocessing
d = preprocessing.normalize(df)

scaled_df = pd.DataFrame(d, columns=df.columns)
scaled_df.head()


# In[13]:


df.columns


# In[14]:


features=list(df.columns[1:31])


# In[15]:


corr= df[features].corr()


# In[16]:


corr


# ## Training the model

# In[17]:




#choosen_features=['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean','radius_worst','perimeter_worst','area_worst','compactness_worst','concavity_worst','concave points_worst']


# In[105]:


from sklearn.model_selection import train_test_split
x=df[features]
y=df['diagnosis']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)


# In[76]:





# In[77]:





# In[78]:





# In[79]:





# In[23]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[106]:


print("Shape of training data is: ", x_train.shape)
print("Shape of testing data is: ", x_test.shape)


# In[107]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
#x_train2=scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[108]:


#x_train = x_train.reshape(455,30,1)
#x_test= x_test.reshape(114,30,1)


# In[109]:


x_train


# In[110]:


from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores ": results['train_accuracy'],
 
              "Training Precision scores ": results['train_precision'],
     
       
              "Validation Accuracy scores ": results['test_accuracy'],
              "Validation Precision scores ": results['test_precision'],

  
              }


# In[111]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D

from keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv1D,MaxPool1D
def create_model():
 model = Sequential()

 model.add(Conv1D(30,5,input_shape=(30,1),activation='relu'))
 model.add(MaxPool1D(pool_size=2))
 #model.add(BatchNormalization())
 model.add(Dropout(0.2))
 model.add(Conv1D(60,5,activation='relu'))
 model.add(MaxPool1D(pool_size=2)) 
 #model.add(BatchNormalization())

 model.add(Dropout(0.2))

 model.add(Flatten())
 model.add(Dense(64,activation='relu'))
 
 model.add(Dense(units=1,activation='sigmoid'))

 model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#  model = Sequential()

#  model.add(Conv1D(32,5,input_shape=(30,1),activation='relu'))
#  model.add(MaxPool1D(pool_size=3))
#  model.add(Dropout(0.5))
#  model.add(Conv1D(64,5,activation='relu'))
#  model.add(MaxPool1D(pool_size=3))

#  model.add(Dropout(0.5))

#  model.add(Flatten())
#  model.add(Dense(units=1,activation='sigmoid'))

#  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 return model



# In[112]:


model=create_model()


# In[197]:


model.summary()
plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)


# In[114]:


model_as_file = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)


# In[115]:



pat=5
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)


# In[116]:


def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=128):
    
    model = create_model()
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_as_file], 
              verbose=1, validation_split=0.1)  
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results 


# In[207]:


# n_folds=3
# epochs=200
# batch_size=128

# #save the model history in a list after fitting so that we can plot later
# model_history = [] 

# for i in range(n_folds):
#     print("Training on Fold: ",i+1)
#     X_train,X_test,Y_train,Y_test  = train_test_split(x_train,y_train, test_size=0.33, 
#                                                random_state = np.random.randint(1,1000, 1)[0])
#     model_history.append(fit_and_evaluate(X_train,X_test,Y_train,Y_test, epochs, batch_size))
#     print("======="*12, end="\n\n\n")


# In[118]:


# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# from keras.callbacks import EarlyStopping
# overfitCallback = EarlyStopping(monitor='loss', patience = 3)
# model = KerasClassifier(model=create_model, epochs=10000, batch_size=10, verbose=0,callbacks=[overfitCallback])
# # evaluate using 03-fold cross validation
# kfold = 3
# accuracy = cross_val_score(model, x_train, y_train, cv=kfold,scoring='accuracy')
# print('accuracy:')
# print(accuracy.mean())
# # recall = cross_val_score(model, x_train, y_train, cv=kfold,scoring='recall')
# # print('recall:')
# # print(recall.mean())
# # precision = cross_val_score(model, x_train, y_train, cv=kfold,scoring='precision')
# # print('precision:')
# # print(precision.mean())
# # f1_macro = cross_val_score(model, x_train, y_train, cv=kfold,scoring='f1_macro')
# # print('f1_macro:')
# # print(f1_macro.mean())
# # model=create_model()
# history = model.fit(x_train,y_train)

# # #results2= cross_validation(model,x_train,y_train,cv=kfold)


# In[208]:


# plt.title('Accuracies vs Epochs')
# plt.plot(model_history[0].history['accuracy'], label='Training Fold 1')
# plt.plot(model_history[1].history['accuracy'], label='Training Fold 2')
# plt.plot(model_history[2].history['accuracy'], label='Training Fold 3')
# plt.plot(model_history[0].history['val_accuracy'], label='validation Fold 1')
# plt.plot(model_history[1].history['val_accuracy'], label='validation Fold 2')
# plt.plot(model_history[2].history['val_accuracy'], label='validation Fold 3')
# plt.legend()
# plt.show()


# In[120]:


model = load_model('model.h5')


# In[121]:


c=model.evaluate(x_test, y_test)


# In[173]:


test_preds = (model.predict(x_test)>0.5 ).astype("int32")
print(test_preds)


# In[191]:


y_pred = (model.predict(np.array(x_test[3],ndmin=2))>0.5).astype("int32")
if y_pred[0] == 0:
   print('benign')
else:
    print('malignant')


# In[212]:






    
def predict_prob(number):
  return [number[0],1-number[0]]

y_prob = np.array(list(map(predict_prob, model.predict(np.array(x_test[random.randint(0,113)],ndmin=2)))))
if y_prob[0][0] > 0.5 :
    print('benign')
else:
    print('malignant')
y_prob 


# In[ ]:





# In[256]:






# In[257]:





# In[ ]:





# In[ ]:





# In[202]:


df['diagnosis'].value_counts()


# In[203]:



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




