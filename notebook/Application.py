#!/usr/bin/env python
# coding: utf-8

# In[2]:


from crypt import methods
from glob import glob
from time import clock_getres

from django.shortcuts import render
from sqlalchemy import false
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



# from UbreastCancer import *

from UPLUSbreastCancer import *
# In[ ]:




from flask import Flask, g,request, url_for, redirect, render_template
import pickle
import numpy as np

import random

from flask import Flask, render_template, url_for, redirect,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

from werkzeug.security import generate_password_hash, check_password_hash





app = Flask(__name__,template_folder='template')
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test1.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
model=pickle.load(open('model.pkl','rb'))#import pickle
UPLOAD_FOLDER = '/home/acer/Documents/notebook'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'csv'}
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'











@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    doctor = db.Column(db.Boolean, default=False)
    Admin = db.Column(db.Boolean, default=False)
    patient = db.Column(db.Boolean, default=True)
    activate= db.Column(db.Boolean,default=True)


db.create_all()

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    
    submit = SubmitField('register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('homm.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         user = User.query.filter_by(username=form.username.data).first()
#         if user:
#             if bcrypt.check_password_hash(user.password, form.password.data):
#                 login_user(user)
#                 return render_template('app.html', form=form)
#     return render_template('login.html', form=form)


# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def dashboard():
#     return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return render_template("homm.html")




@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    # if form.validate_on_submit():
    #     hashed_password = bcrypt.generate_password_hash(form.password.data)
    #     new_user = User(username=form.username.data, password=hashed_password)
    #     db.session.add(new_user)
    #     db.session.commit()
    #     return redirect(url_for('login'))

    # return render_template('register.html', form=form)
    username = request.form.get("username", False)
     
    password = request.form['password']
    new_user = User(username=username, password=form.password.data)
    db.session.add(new_user)
    db.session.commit()

     

    return   render_template("homm.html")
    






@app.route('/app')
@login_required
def hello_world():
    return render_template("app.html")

# print("~<HOf!Csj")
@app.route('/predict',methods=['POST','GET'])
def predictt():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features,ndmin=2)]
    print(int_features)
    print(final)
    y_prob = np.array(list(map(predict_prob, model.predict(final))))


    if y_prob[0][0]>0.5:
        return render_template('app.html',pred='Your health is Safe.\nProbability of benign occuring is {}'.format(y_prob[0][0]),bhai="kuch karna hain iska ab?")
    else:
        return render_template('app.html',pred='Your health is in, Danger.\n Probability of malignant cell occuring is {}'.format(y_prob[0][1]),bhai="Your Forest is Safe for now")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('app.html',pred='Your health is safe.\n Probability of benign occuring is {}'.format(prediction[1]),bhai="Your Forest is Safe for now")


def RandomizeVEct():
  v=np.array(x_test[random.randint(0, 113)],ndmin=2)
  return v
@app.route('/uploadd',methods=['POST','GET'])
def predictDB():
    global v
    v=RandomizeVEct()
    
    
    print(v[0])

    #output='{0:.{1}f}'.format(prediction[0][1], 2)

   
    return render_template('uploaded.html',v=v)


@app.route('/predictFromDB',methods=['POST','GET'])
def predictt2():

   
    # prediction=model.predict_proba(v)
    # #output='{0:.{1}f}'.format(prediction[0][1], 2)
    # return render_template('res.html',pred="the probability of having cancer is {}%".format(prediction[0]*100))
    y_prob = np.array(list(map(predict_prob, model.predict(v))))

    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if y_prob[0][0]>0.5:
        return render_template('app3.html',pred='Your health is Safe.\nProbability of benign occuring is {}'.format(y_prob[0][0]*100),bhai="kuch karna hain iska ab?")
    else:
        return render_template('app3.html',pred='Your health is in, Danger.\n Probability of malignant cell occuring is {}'.format(y_prob[0][1]*100),bhai="Your Forest is Safe for now")

    # if prediction[0]>0.5:
    #     return render_template('res.html',pred='Your health is in Danger.\nProbability of Malignant occuring is {} %'.format(prediction[0]*100),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('res.html',pred='Your health is safe.\n Probability of benign occuring is {} %'.format(prediction[1]*100),bhai="Your Forest is Safe for now")

from sqlalchemy.sql import text


@app.route('/update/<id>',methods=['GET','POST'])
def update1(id):
    is_checked=request.form.get('Activated')
    print(id)
    is_checked2=request.form.get('Desactivated')    

    if is_checked!=None:
#             print("None")
#             user = db.engine.execute(
#      text("""UPDATE user
# SET activate = True
# WHERE id = id;"""),id=id,activate=True)
#     message="user activated "
      user = User.query.filter_by(id=id).first()
      print(id)
      user.activate=1
      db.session.commit()
      message="user activated "
    elif is_checked2!=None:
      user = User.query.filter_by(id=id).first()
      user.activate=0
      db.session.commit()
      message="user desactivated "
    return render_template("patient.html",message=message)
    #  print(id)
    #  print(a)
    #  print(user)
     
    #  u1= db.engine.execute(text("SElECT * from user where id=id;"),id=id)

   
          
    #  u1.activate = a
    #  print(u1.activate)
               
     
    




@app.route('/doctorsList', methods=['GET', 'POST'])
@login_required

def doctorsList():
  #   loginadmin()
     rowss = db.session.query(User.doctor==True,User.Admin==0).count()
     global doctors
     doctors = User.query.filter_by(Admin=False,doctor=True).all()
     owner = User.query.filter_by(username=current_user.username,password=current_user.password,Admin=True).first()
     if not owner:

          flash("The logins provided is not an admin!")
          return redirect(url_for('home'))
     
     else:

          return render_template('patient.html' ,patients = doctors,rowss=rowss)





@app.route('/patients', methods=['GET', 'POST'])
@login_required

def patients():
  #   loginadmin()
     rows = db.session.query(User.doctor==0,User.Admin==0).count()
     global user_patients
     
     user_patients = User.query.filter_by(Admin=False,doctor=False).all()
     owner = User.query.filter_by(username=current_user.username,password=current_user.password,Admin=True).first()
     if not owner:

          flash("The logins provided is not an admin!")
          return redirect(url_for('home'))
     
     else:

          return render_template('patient.html' ,patients = user_patients,rows=rows)

@app.route('/doctors',methods=['GET', 'POST'])
@login_required

def doctors():
  #   loginadmin()
     
     user_doctors = User.query.filter_by(doctor=True).all()
     owner = User.query.filter_by(username=current_user.username,password=current_user.password,Admin=True).first()
     if not owner:

          flash("The logins provided is not an admin!")
          return redirect(url_for('home'))
     
     else:

          return render_template('doctor.html' ,doctors = user_doctors)


@app.route('/register_doctor',methods = ['POST','GET'])
def register_doctor():

     form = RegisterForm()

    
     username = request.form.get("username", False)

     
     password = request.form['password']
     new_user = User(username=username, password=form.password.data,doctor=True,patient=False)
     db.session.add(new_user)
     db.session.commit()
     return   render_template('admin.html')
@app.route('/loginpatient',methods = ['POST','GET'])
def loginpatient ():
     username = request.form['username']
     
     password = request.form['password']
     owner = User.query.filter_by(username=username).first()
     if owner.activate == False:
        flash("Username or password is wrong")
        print("you are wrong")
        return   render_template('homm.html',pr="your account was desactivated")
     if owner.password != password:
          flash("Username or password is wrong")
          print("you are wrong")

          return   redirect(url_for('home'))
     elif owner.username == "admin1":
      if owner.password == password:
          login_user(owner)
          flash("Welcome")
          print("you are write")
          return   render_template('admin.html')
     elif owner.doctor == False:
      if owner.password == password:
          login_user(owner)
          return   render_template('app2.html')

     elif owner.doctor == True:
      if owner.password == password:
          login_user(owner)
          return   render_template('doctor_interface.html')

     else: return   render_template('user.html')
     






@app.route('/testingDB',methods=['POST','GET'])
def testingDb():
    
    return render_template('res.html',pred="the accuracy of tested portion is {}%".format(c[1]*100))


# @app.route("/messg" ,methods= ['POST','GET'])
# def mymessage():

#     em = request.form['email']
#     mm = request.form['message']
#     msg = Message('Hello', sender = 'jxkalmhefacbuk@gmail.com', recipients = [em])
#     msg.body = mm
#     mail.send(msg)
#     flash("Message sent successfully") 
#     return redirect(url_for('get_appointment_recieved'))


if __name__ == '__main__':
    app.run(port=8099,debug=True)


# In[9]:





# In[15]:





# In[3]:


from pywebio import start_server

from pywebio.input import *
from pywebio.output import *

def bmicalculator():
    radius_mean=input("Please enter the radius mean",type=FLOAT)
    #texture_mean=input("Please enter the value",type=FLOAT)
    perimeter_mean=input("Please enter the perimeter mean",type=FLOAT)
    area_mean=input("Please enter the area mean",type=FLOAT)
    #smoothness_mean=input("Please enter the value",type=FLOAT)
    #compactness_mean=input("Please enter the value",type=FLOAT)
    #concavity_mean=input("Please enter the value",type=FLOAT)
    concavepoints_mean=input("Please enter the concave points mean",type=FLOAT)
    #symmetry_mean=input("Please enter the value",type=FLOAT)
    #fractal_dimension_mean=input("Please enter the value",type=FLOAT)
    #radius_se=input("Please enter the value",type=FLOAT)
    #texture_se=input("Please enter the value",type=FLOAT)
    #perimeter_se=input("Please enter the value",type=FLOAT)
    #area_se=input("Please enter the value",type=FLOAT)
    #smoothness_se=input("Please enter the value",type=FLOAT)
    #compactness_se=input("Please enter the value",type=FLOAT)
    #concavity_se=input("Please enter the value",type=FLOAT)
    #concavepoints_se=input("Please enter the value",type=FLOAT)
    #symmetry_se=input("Please enter the value",type=FLOAT)
    #fractal_dimension_se=input("Please enter the value",type=FLOAT)
    radius_worst=input("Please enter the radius worst",type=FLOAT)
    #texture_worst=input("Please enter the value",type=FLOAT)
    perimeter_worst=input("Please enter the perimeter worst",type=FLOAT)
    area_worst=input("Please enter the area worst",type=FLOAT)
    #smoothness_worst=input("Please enter the value",type=FLOAT)
    #compactness_worst=input("Please enter the value",type=FLOAT)
    #concavity_worst=input("Please enter the value",type=FLOAT)
    concavepoints_worst=input("Please enter the concave points worst",type=FLOAT)
    #symmetry_worst=input("Please enter the  value",type=FLOAT)
    #fractal_dimension_worst=input("Please enter the value",type=FLOAT)
    
    import numpy as np

    from sklearn import preprocessing
    
    a=np.array([radius_mean,
 0,
 perimeter_mean,
 area_mean,
 0,
 0,
 0,
 concavepoints_mean,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 radius_worst,
 0,
 perimeter_worst,
 area_worst,
 0,
 0,
 0,
    concavepoints_worst,
    0,
    0],ndmin=2)
    min_max_scaler = preprocessing.MinMaxScaler()

    a = min_max_scaler.fit_transform(a)
    prediction=pmodel.predict(a)
    p=pmodel.predict_proba(a)
    if prediction==[0]:
        put_text('Your BMI is  : B')
    else:
        put_text('Your BMI is  : M')
    put_text('the score is:'% p) 

        
#start_server(bmicalculator,port=4767,debug=True)


# In[ ]:





# In[ ]:




