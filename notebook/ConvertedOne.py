#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


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


# In[18]:


from sklearn.model_selection import train_test_split
#train,test = train_test_split(df, test_size=0.0, random_state=1)


# In[19]:


x_train=df[features]
#x_train2=train[features]


# In[20]:


y_train=df['diagnosis']
#y_train2=train['diagnosis']


# In[21]:


#x_test=test[features]


# In[22]:


#y_test=test['diagnosis']


# In[23]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[24]:


print("Shape of training data is: ", x_train.shape)
#print("Shape of testing data is: ", x_test.shape)


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
#x_train2=scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)


# In[26]:


#x_train = x_train.reshape(455,30,1)
#x_test= x_test.reshape(114,30,1)


# In[27]:


x_train


# In[28]:


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


# In[29]:


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
 model.add(Dense(units=1,activation='sigmoid'))

 model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 return model



# In[ ]:





# In[ ]:





# In[30]:


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
overfitCallback = EarlyStopping(monitor='loss', patience = 3)
model = KerasClassifier(model=create_model, epochs=10000, batch_size=10, verbose=0,callbacks=[overfitCallback])
model=model.fit(x_train,y_train)
# evaluate using 03-fold cross validation
kfold = 3
accuracy = cross_val_score(model, x_train, y_train, cv=kfold,scoring='accuracy')
print('accuracy:')
print(accuracy.mean())
recall = cross_val_score(model, x_train, y_train, cv=kfold,scoring='recall')
print('recall:')
print(recall.mean())
precision = cross_val_score(model, x_train, y_train, cv=kfold,scoring='precision')
print('precision:')
print(precision.mean())
f1_macro = cross_val_score(model, x_train, y_train, cv=kfold,scoring='f1_macro')
print('f1_macro:')
print(f1_macro.mean())

#results2= cross_validation(model,x_train,y_train,cv=kfold)


# In[31]:


from tensorflow.keras.utils import plot_model
m=create_model() # `input_shape` is the shape of the input data
                         # e.g. input_shape = (None, 32, 32, 3)

m.summary()
plot_model(m, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)


# In[32]:


import pickle
#pickle.dump(model, open('model.pkl', 'wb')) #for saving
m.save("my_model")



# In[ ]:





# In[33]:


import seaborn as sns
sns.lineplot(data=accuracy)
sns.lineplot(data=recall)
sns.lineplot(data=precision)
sns.lineplot(data=f1_macro)


# In[ ]:





# In[34]:


df['diagnosis'].value_counts()


# In[35]:


sns.countplot(x="diagnosis", data=df)


# In[36]:


from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
cv=CountVectorizer()
sample_text =pd.Series([0.23, 0.13,0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03])
ar=np.array(sample_text)


# In[37]:


X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.3, random_state=42)


# In[38]:


from sklearn.metrics import plot_confusion_matrix

model.predict(x_train)
model.score(x_train,y_train)
plot_confusion_matrix(model,x_train,y_train)


# In[39]:


a=model.predict(x_train)
model.score(x_train,y_train)


# In[ ]:





# In[40]:


yt=df.iloc[1]
yt.drop
yt.drop('diagnosis')
proba = model.predict(x_train, batch_size=1)
print(proba)


# In[41]:




aa=model.predict(np.array(X_test[0],ndmin=2))
ss=np.array([0.23, 0.13,0, 0.39, 0, 0.4, 0.07, 0.1, 0.0, 0.03, 0.23, 0, 0.43, 0, 0.88, 0, 0, 0, 0.06, 0, 0, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03],ndmin=2)
print(aa)
if aa==[0]:
    print("Benign")
model.predict(ss)
model.predict_proba(ss)


# In[42]:


X_test[1]


# In[43]:


np.array([0.23, 0.13,0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03],ndmin=2)


# In[44]:


sampl = np.random.uniform(low=0, high=1, size=(30))


# In[45]:


sampl


# In[46]:


features


# In[47]:


def Testing(radius_mean,
 texture_mean,
 perimeter_mean,
 area_mean,
 smoothness_mean,
 compactness_mean,
 concavity_mean,
 concavepoints_mean,
 symmetry_mean,
 fractal_dimension_mean,
 radius_se,
 texture_se,
 perimeter_se,
 area_se,
 smoothness_se,
 compactness_se,
 concavity_se,
 concavepoints_se,
 symmetry_se,
 fractal_dimension_se,
 radius_worst,
 texture_worst,
 perimeter_worst,
 area_worst,
 smoothness_worst,
 compactness_worst,
 concavity_worst,
 concavepoints_worst,
 symmetry_worst,
 fractal_dimension_worst,n=2):
    a=np.array([radius_mean,
 texture_mean,
 perimeter_mean,
 area_mean,
 smoothness_mean,
 compactness_mean,
 concavity_mean,
 concavepoints_mean,
 symmetry_mean,
 fractal_dimension_mean,
 radius_se,
 texture_se,
 perimeter_se,
 area_se,
 smoothness_se,
 compactness_se,
 concavity_se,
 concavepoints_se,
 symmetry_se,
 fractal_dimension_se,
 radius_worst,
 texture_worst,
 perimeter_worst,
 area_worst,
 smoothness_worst,
 compactness_worst,
 concavity_worst,
 concavepoints_worst,
 symmetry_worst,
 fractal_dimension_worst],ndmin=2)
    prediction=model.predict(a)
    result="Benign" if prediction==[0] else "Malignant"
    output="The result is"+result
    return(output,result)


# In[48]:


Testing(0.23, 0.13,0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03, 0.23, 0.13, 0.43, 0.39, 0.88, 0.4, 0.07, 0.1, 0.06, 0.03)


# In[49]:


import gradio as gr


# In[50]:


interface=gr.Interface(fn=Testing,
                       inputs=["text",gr.inputs.Slider(0,1,label="radius_mean"),gr.inputs.Slider(0,1,label="texture_mean"),gr.inputs.Slider(0,1,label="perimeter_mean"),gr.inputs.Slider(0,1,label="area_mean"),gr.inputs.Slider(0,1,label="smoothness_mean"),gr.inputs.Slider(0,1,label="compactness_mean"),gr.inputs.Slider(0,1,label="concavity_mean"),gr.inputs.Slider(0,1,label="concavepoints_mean"),gr.inputs.Slider(0,1,label="symmetry_mean"),gr.inputs.Slider(0,1,label="fractal_dimension_mean"),gr.inputs.Slider(0,1,label="radius_se"),gr.inputs.Slider(0,1,label="texture_se"),gr.inputs.Slider(0,1,label="perimeter_se"),gr.inputs.Slider(0,1,label="area_se"),gr.inputs.Slider(0,1,label="smoothness_se"),gr.inputs.Slider(0,1,label="compactness_se"),gr.inputs.Slider(0,1,label="concavity_se"),gr.inputs.Slider(0,1,label="concavepoints_se"),gr.inputs.Slider(0,1,label="symmetry_se"),gr.inputs.Slider(0,1,label="fractal_dimension_se"),gr.inputs.Slider(0,1,label="radius_worst"),gr.inputs.Slider(0,1,label="texture_worst"),gr.inputs.Slider(0,1,label="perimeter_worst"),gr.inputs.Slider(0,1,label="area_worst"),gr.inputs.Slider(0,1,label="smoothness_worst"),gr.inputs.Slider(0,1,label="compactness_worst"),gr.inputs.Slider(0,1,label="concavity_worst"),gr.inputs.Slider(0,1,label="concavepoints_worst"),gr.inputs.Slider(0,1,label="symmetry_worst"),gr.inputs.Slider(0,1,label="fractal_dimension_worst")],
                      outputs=["text","text"])


# In[51]:


#interface.launch()


# In[52]:


pickle.dump(model,open('model.pkl','wb'))
model1=pickle.load(open('model.pkl','rb'))


# In[73]:


inputt=[float(x) for x in "0.23 0.13 0.43 0.39 0.88 0.4 0.07 0.1 0.06 0.03 0.23 0.13 0.43 0.39 0.88 0.4 0.07 0.1 0.06 0.03 0.23 0.13 0.43 0.39 0.88 0.4 0.07 0.1 0.06 0.03".split(' ')]
final=[np.array(inputt)]

b = model.predict_proba(final)
print(b[0])


# In[ ]:





# In[ ]:


from pywebio import start_server

from pywebio.input import *
from pywebio.output import *

# def bmicalculator():
#     radius_mean=input("Please enter the value",type=FLOAT)
#     #texture_mean=input("Please enter the value",type=FLOAT)
#     perimeter_mean=input("Please enter the value",type=FLOAT)
#     area_mean=input("Please enter the value",type=FLOAT)
#     #smoothness_mean=input("Please enter the value",type=FLOAT)
#     #compactness_mean=input("Please enter the value",type=FLOAT)
#     #concavity_mean=input("Please enter the value",type=FLOAT)
#     #concavepoints_mean=input("Please enter the value",type=FLOAT)
#     #symmetry_mean=input("Please enter the value",type=FLOAT)
#     #fractal_dimension_mean=input("Please enter the value",type=FLOAT)
#     #radius_se=input("Please enter the value",type=FLOAT)
#     #texture_se=input("Please enter the value",type=FLOAT)
#     #perimeter_se=input("Please enter the value",type=FLOAT)
#     #area_se=input("Please enter the value",type=FLOAT)
#     #smoothness_se=input("Please enter the value",type=FLOAT)
#     #compactness_se=input("Please enter the value",type=FLOAT)
#     #concavity_se=input("Please enter the value",type=FLOAT)
#     #concavepoints_se=input("Please enter the value",type=FLOAT)
#     #symmetry_se=input("Please enter the value",type=FLOAT)
#     #fractal_dimension_se=input("Please enter the value",type=FLOAT)
#     radius_worst=input("Please enter the value",type=FLOAT)
#     #texture_worst=input("Please enter the value",type=FLOAT)
#     perimeter_worst=input("Please enter the value",type=FLOAT)
#     area_worst=input("Please enter the value",type=FLOAT)
#     #smoothness_worst=input("Please enter the value",type=FLOAT)
#     #compactness_worst=input("Please enter the value",type=FLOAT)
#     #concavity_worst=input("Please enter the value",type=FLOAT)
#     concavepoints_worst=input("Please enter the value",type=FLOAT)
#     #symmetry_worst=input("Please enter the  value",type=FLOAT)
#     #fractal_dimension_worst=input("Please enter the value",type=FLOAT)
    
#     import numpy as np

#     from sklearn import preprocessing
    
#     a=np.array([radius_mean,
#  0,
#  perimeter_mean,
#  area_mean,
#  0,
#  0,
#  0,
#  concavepoints_mean,
#  0,
#  0,
#  0,
#  0,
#  0,
#  0,
#  0,
#  0,
#  0,
#  concavepoints_se,
#  0,
#  0,
#  radius_worst,
#  0,
#  perimeter_worst,
#  area_worst,
#  0,
#  0,
#  0,
#     concavepoints_worst,
#     0,
#     0],ndmin=2)
#     min_max_scaler = preprocessing.MinMaxScaler()

#     a = min_max_scaler.fit_transform(a)
#     prediction=model.predict(a)
#     if prediction==[0]:
#         put_text('Your BMI is  : B')
#     else:
#         put_text('Your BMI is  : M')
        

        
        

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np



from flask import Flask, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt



app=Flask(__name__,template_folder='template')

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
model=pickle.load(open('model.pkl','rb'))#import pickle

@app.route('/')
def hello_world():
    return render_template("app.html")


@app.route('/predict',methods=['POST','GET'])
def predictt():
    int_features=[int(x) for x in request.form.values()]
    print(request.form.values())
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction[0]>0.5:
        return render_template('app.html',pred='Your health is in Danger.\nProbability of malignant is {}'.format(prediction[0]),bhai="")
    else:
        return render_template('app.html',pred='Your health is safe.\n Probability of benign is {}'.format(prediction[1]),bhai="")


if __name__ == '__main__':
    app.run()
    
    
    
    





# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))


# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(20), nullable=False, unique=True)
#     password = db.Column(db.String(80), nullable=False)


# class RegisterForm(FlaskForm):
#     username = StringField(validators=[
#                            InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

#     password = PasswordField(validators=[
#                              InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

#     submit = SubmitField('Register')

#     def validate_username(self, username):
#         existing_user_username = User.query.filter_by(
#             username=username.data).first()
#         if existing_user_username:
#             raise ValidationError(
#                 'That username already exists. Please choose a different one.')


# class LoginForm(FlaskForm):
#     username = StringField(validators=[
#                            InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

#     password = PasswordField(validators=[
#                              InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

#     submit = SubmitField('Login')


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         user = User.query.filter_by(username=form.username.data).first()
#         if user:
#             if bcrypt.check_password_hash(user.password, form.password.data):
#                 login_user(user)
#                 return redirect(url_for('dashboard'))
#     return render_template('login.html', form=form)


# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def dashboard():
#     return render_template('dashboard.html')


# @app.route('/logout', methods=['GET', 'POST'])
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


# @ app.route('/register', methods=['GET', 'POST'])
# def register():
#     form = RegisterForm()

#     if form.validate_on_submit():
#         hashed_password = bcrypt.generate_password_hash(form.password.data)
#         new_user = User(username=form.username.data, password=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         return redirect(url_for('login'))

#     return render_template('register.html', form=form)


# if __name__ == "__main__":
#     app.run(debug=True)
#start_server(bmicalculator,port=4767,debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




