# from django.shortcuts import render

# from django.http import HttpResponse
# from .models import *
# import pandas as pd
# #import pickel
# import joblib
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder,StandardScaler





# # Create your views here.

# #Initializing empty dataframe for global activities
# dfm=pd.DataFrame()
# X_traing=pd.DataFrame()
# X_testg=pd.DataFrame()
# y_traing=pd.DataFrame()
# y_testg=pd.DataFrame()

# #Function for creating dataframe from file

# def create_df(file_path):
#     df=pd.read_csv(file_path,delimiter=',')
#     return df


# #DataPreprocessing Function for Classification

# def preprocessing(df):
#     for i in df:
#         i=str(i)
#         if(df[i].isna().sum()>0):
#             if(isinstance(df[i][0],str)):
#                 df[i]=df[i].fillna(df[i].mode()[0])
#             else:
#                 df[i]=df[i].fillna(df[i].mean())
#     le=LabelEncoder()            
#     for i in df:
#         i=str(i)
#         if(isinstance(df[i][0],str)):
#             df[i]=le.fit_transform(df[i])
            
#     cols=len(df.columns)
#     X=df.iloc[:,0:cols-1]
#     y=df.iloc[:,cols-1:]
    
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    
#     return X_train,X_test,y_train,y_test

# #DataPreprocessing Function for SVR

# def preprocessing_svr(df):
#     for i in df:
#         i=str(i)
#         if(df[i].isna().sum()>0):
#             if(isinstance(df[i][0],str)):
#                 df[i]=df[i].fillna(df[i].mode()[0])
#             else:
#                 df[i]=df[i].fillna(df[i].mean())
                
                
#     le=LabelEncoder()            
#     for i in df:
#         i=str(i)
#         if(isinstance(df[i][0],str)):
#             df[i]=le.fit_transform(df[i])
    
    
#     cols=len(df.columns)
#     X=df.iloc[:,0:cols-1]
#     y=df.iloc[:,cols-1:]
    
#     sc=StandardScaler()
#     X_=sc.fit_transform(X)
#     y_=sc.fit_transform(y)
    
    
    
#     X_train,X_test,y_train,y_test=train_test_split(X_,y_,test_size=0.2,random_state=2)
    
#     return X_train,X_test,y_train,y_test

# #Result SVR

# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# def s_v_r(n,X_train,X_test,y_train,y_test):
#     svr=SVR()
#     svr.fit(X_train,y_train)
#     y_pred=svr.predict(X_test)
    
#     # print(mean_squared_error(y_test,y_pred))#MSE
#     # print(mean_squared_error(y_test,y_pred,squared=False))#RMSE
#     # print(mean_absolute_error(y_test,y_pred))#MAE
#     # print(r2_score(y_test,y_pred))#R2

#     mse=mean_squared_error(y_test,y_pred)
#     rmse=mean_squared_error(y_test,y_pred,squared=False)
#     mae=mean_absolute_error(y_test,y_pred)
#     r2=r2_score(y_test,y_pred)

#     return mse,rmse,mae,r2
# #DataPreprocessing Function for Linear Regression

# def preprocessing_lr(df):
#     k_=list(df)
#     class_label=k_[len(k_)-1]
#     for i in df:
#         i=str(i)
#         if(df[i].isna().sum()>0):
#             if(isinstance(df[i][0],str)):
#                 df[i]=df[i].fillna(df[i].mode()[0])
#             else:
#                 df[i]=df[i].fillna(df[i].mean())
                
#     cols=len(df.columns)
    
#     le=LabelEncoder()            
#     for i in df:
#         i=str(i)
#         if(isinstance(df[i][0],str)):
#             df[i]=le.fit_transform(df[i])
    
#     for i in df:
#         i=str(i)
#         c=np.corrcoef(df[i],df[class_label])
#         if(c[1][0]<0.4):
#             df=df.drop([i],axis=1)
    
#     X=df.iloc[:,0:cols-1]
#     y=df[class_label]
    
#     #sc=StandardScaler()
#     #X_=sc.fit_transform(X)
#     #y_=sc.fit_transform(y)
    
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
    
#     return X_train,X_test,y_train,y_test

# # Linear Regression Result
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# def linearRegression(n,X_train,X_test,y_train,y_test):
    
#     lr=LinearRegression()
#     lr.fit(X_train,y_train)
#     y_pred=lr.predict(X_test)
    
#     # print(mean_squared_error(y_test,y_pred))#MSE
#     # print(mean_squared_error(y_test,y_pred,squared=False))#RMSE
#     # print(mean_absolute_error(y_test,y_pred))#MAE
#     # print(r2_score(y_test,y_pred))#R2
    
#     mse=mean_squared_error(y_test,y_pred)
#     rmse=mean_squared_error(y_test,y_pred,squared=False)
#     mae=mean_absolute_error(y_test,y_pred)
#     r2=r2_score(y_test,y_pred)

#     return mse,rmse,mae,r2

# #Algorithms based on options

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
# def mlmodel(n,X_train,X_test,y_train,y_test):
#     n=int(n)
#     if(n==1):
#         dt=DecisionTreeClassifier(criterion="entropy")
#         dt.fit(X_train,y_train)
#         #joblib.dump(dt,)
#         y_pred_test=dt.predict(X_test)
#         y_pred_train=dt.predict(X_train)
        
#     elif(n==2):
#         lr=LogisticRegression()
#         lr.fit(X_train,y_train)
#         y_pred_test=lr.predict(X_test)
#         y_pred_train=lr.predict(X_train)
    
#     elif(n==3):
#         gnb = GaussianNB()
#         gnb.fit(X_train,y_train)
#         y_pred_train=gnb.predict(X_train)
#         y_pred_test=gnb.predict(X_test)
        
#     elif(n==4):
#         neigh = KNeighborsClassifier(n_neighbors=3)
#         neigh.fit(X_train,y_train)
#         y_pred_train=neigh.predict(X_train)
#         y_pred_test=neigh.predict(X_test)
        
#     elif(n==5):
#         sv=SVC()
#         sv.fit(X_train,y_train)
#         y_pred_train=sv.predict(X_train)
#         y_pred_test=sv.predict(X_test)
        
#     accuracy=accuracy_score(y_test,y_pred_test)
#     precision=precision_score(y_test,y_pred_test,zero_division=1)
#     recall=recall_score(y_test,y_pred_test,zero_division=1)
#     f1=f1_score(y_test,y_pred_test,zero_division=1)
#     cm=confusion_matrix(y_test,y_pred_test)
    
#     return accuracy,precision,recall,f1,cm




# #Function called when home url is called

# def home(request):
#     if request.method == "POST":
#         file = request.FILES['file']
#         obj = File.objects.create(file=file)
#         df=create_df(obj.file)
#         global dfm
#         dfm=df
#         # global dfm,X_traing,X_testg,y_traing,y_testg
#         # X_train,X_test,y_train,y_test=preprocessing(df)
#         # dfm=df
#         # X_traing=X_train
#         # X_testg=X_test
#         # y_traing=y_train
#         # y_testg=y_test


#         #print(X_train)
        
#         # return render(request,'response.html',{'val':1,'df':df,'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test})
#         return render(request,'response.html')
#     else:
#         return render(request,'home.html',{'name':'abcde'})

# def response(request):
#     return render(request,'response.html')

# def abcd(request):
#     #print(dfm)
#     option=request.POST['option']
#     print(option)
#     option=int(option)
#     if(option>=0 and option<=5):
#         X_train,X_test,y_train,y_test=preprocessing(dfm)
#         accuracy,precision,recall,f1,cm=mlmodel(option,X_train,X_test,y_train,y_test)
#         return render(request,'temp.html',{'option':option,'df':dfm,'range':range(len(dfm)),'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'cm':cm})
    
#     if(option==6):
#         X_train,X_test,y_train,y_test=preprocessing_lr(dfm)
#         mse,rmse,mae,r2=linearRegression(option,X_train,X_test,y_train,y_test)
#         return render(request,'temp.html',{'option':option,'df':dfm,'mse':mse,'rmse':rmse,'mae':mae,'r2':r2})
    
#     if(option==7):
#         X_train,X_test,y_train,y_test=preprocessing_svr(dfm)
#         mse,rmse,mae,r2=s_v_r(option,X_train,X_test,y_train,y_test)
#         return render(request,'temp.html',{'option':option,'df':dfm,'mse':mse,'rmse':rmse,'mae':mae,'r2':r2})





# def index(request):
#     return render(request,'index.html',{'val':1})

# def contact(request):
#     return render(request,"aboutus.html")

# def aboutpyfit(request):
#     return render(request,"aboutpyfit.html")



from django.shortcuts import render

from django.http import HttpResponse
from .models import *
import pandas as pd
#import pickel
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Create your views here.

#Initializing empty dataframe for global activities
dfm=pd.DataFrame()
X_traing=pd.DataFrame()
X_testg=pd.DataFrame()
y_traing=pd.DataFrame()
y_testg=pd.DataFrame()

#Function for creating dataframe from file

def create_df(file_path):
    df=pd.read_csv(file_path,delimiter=',')
    return df


#DataPreprocessing Function for Classification

def preprocessing(df):
    for i in df:
        i=str(i)
        if(df[i].isna().sum()>0):
            if(isinstance(df[i][0],str)):
                df[i]=df[i].fillna(df[i].mode()[0])
            else:
                df[i]=df[i].fillna(df[i].mean())
    le=LabelEncoder()            
    for i in df:
        i=str(i)
        if(isinstance(df[i][0],str)):
            df[i]=le.fit_transform(df[i])
            
    cols=len(df.columns)
    X=df.iloc[:,0:cols-1]
    y=df.iloc[:,cols-1:]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    
    return X_train,X_test,y_train,y_test

#DataPreprocessing Function for SVR

def preprocessing_svr(df):
    for i in df:
        i=str(i)
        if(df[i].isna().sum()>0):
            if(isinstance(df[i][0],str)):
                df[i]=df[i].fillna(df[i].mode()[0])
            else:
                df[i]=df[i].fillna(df[i].mean())
                
                
    le=LabelEncoder()            
    for i in df:
        i=str(i)
        if(isinstance(df[i][0],str)):
            df[i]=le.fit_transform(df[i])
    
    
    cols=len(df.columns)
    X=df.iloc[:,0:cols-1]
    y=df.iloc[:,cols-1:]
    
    sc=StandardScaler()
    X_=sc.fit_transform(X)
    y_=sc.fit_transform(y)
    
    
    
    X_train,X_test,y_train,y_test=train_test_split(X_,y_,test_size=0.2,random_state=2)
    
    return X_train,X_test,y_train,y_test

#Result SVR

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def s_v_r(n,X_train,X_test,y_train,y_test):
    svr=SVR()
    svr.fit(X_train,y_train)
    joblib.dump(svr,"media/models_created/"+file_name+".joblib")
    y_pred=svr.predict(X_test)
    
    # print(mean_squared_error(y_test,y_pred))#MSE
    # print(mean_squared_error(y_test,y_pred,squared=False))#RMSE
    # print(mean_absolute_error(y_test,y_pred))#MAE
    # print(r2_score(y_test,y_pred))#R2

    mse=mean_squared_error(y_test,y_pred)
    rmse=mean_squared_error(y_test,y_pred,squared=False)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    return mse,rmse,mae,r2
#DataPreprocessing Function for Linear Regression

def preprocessing_lr(df):
    k_=list(df)
    class_label=k_[len(k_)-1]
    for i in df:
        i=str(i)
        if(df[i].isna().sum()>0):
            if(isinstance(df[i][0],str)):
                df[i]=df[i].fillna(df[i].mode()[0])
            else:
                df[i]=df[i].fillna(df[i].mean())
                
    cols=len(df.columns)
    
    le=LabelEncoder()            
    for i in df:
        i=str(i)
        if(isinstance(df[i][0],str)):
            df[i]=le.fit_transform(df[i])
    
    for i in df:
        i=str(i)
        c=np.corrcoef(df[i],df[class_label])
        if(c[1][0]<0.4):
            df=df.drop([i],axis=1)
    
    X=df.iloc[:,0:cols-1]
    y=df[class_label]
    
    #sc=StandardScaler()
    #X_=sc.fit_transform(X)
    #y_=sc.fit_transform(y)
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
    
    return X_train,X_test,y_train,y_test

# Linear Regression Result
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def linearRegression(n,X_train,X_test,y_train,y_test):
    
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    joblib.dump(lr,"media/models_created/"+file_name+".joblib")
    y_pred=lr.predict(X_test)
    
    # print(mean_squared_error(y_test,y_pred))#MSE
    # print(mean_squared_error(y_test,y_pred,squared=False))#RMSE
    # print(mean_absolute_error(y_test,y_pred))#MAE
    # print(r2_score(y_test,y_pred))#R2
    
    mse=mean_squared_error(y_test,y_pred)
    rmse=mean_squared_error(y_test,y_pred,squared=False)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    return mse,rmse,mae,r2

#Algorithms based on options

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix



def mlmodel(n,X_train,X_test,y_train,y_test):
    n=int(n)
    global file_name
    #str="C:\Users\Uday\Desktop\PyFit_Project\media\models_created\abc"
    if(n==1):
        dt=DecisionTreeClassifier(criterion="entropy")
        dt.fit(X_train,y_train)
        joblib.dump(dt,"media/models_created/"+file_name+".joblib")
        y_pred_test=dt.predict(X_test)
        y_pred_train=dt.predict(X_train)
        
    elif(n==2):
        lr=LogisticRegression()
        lr.fit(X_train,y_train)
        joblib.dump(lr,"media/models_created/"+file_name+".joblib")
        y_pred_test=lr.predict(X_test)
        y_pred_train=lr.predict(X_train)
    
    elif(n==3):
        gnb = GaussianNB()
        gnb.fit(X_train,y_train)
        joblib.dump(gnb,"media/models_created/"+file_name+".joblib")
        y_pred_train=gnb.predict(X_train)
        y_pred_test=gnb.predict(X_test)
        
    elif(n==4):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train,y_train)
        joblib.dump(neigh,"media/models_created/"+file_name+".joblib")
        y_pred_train=neigh.predict(X_train)
        y_pred_test=neigh.predict(X_test)
        
    elif(n==5):
        sv=SVC()
        sv.fit(X_train,y_train)
        joblib.dump(sv,"media/models_created/"+file_name+".joblib")
        y_pred_train=sv.predict(X_train)
        y_pred_test=sv.predict(X_test)
        
    accuracy=accuracy_score(y_test,y_pred_test)
    precision=precision_score(y_test,y_pred_test,zero_division=1)
    recall=recall_score(y_test,y_pred_test,zero_division=1)
    f1=f1_score(y_test,y_pred_test,zero_division=1)
    cm=confusion_matrix(y_test,y_pred_test)
    
    return accuracy,precision,recall,f1,cm


import glob
import os



#Function called when home url is called
file_name=""    
def home(request):
    if request.method == "POST":
        file = request.FILES['file']
        obj = File.objects.create(file=file)
        df=create_df(obj.file)
        global dfm
        global file_name
        dfm=df
        
        # Code for finding file name 
        
        list_of_files = glob.glob('C:/Users/manvi/OneDrive/Desktop/PyFit_Project/media/files/*') # * means all if need specific format then *.csv
        if list_of_files:
            file_path = max(list_of_files, key=os.path.getctime)
        else:
            file_path=0
        file_name=os.path.splitext(str(file_path))
        file_name=file_name[0]
        temp=file_name.split('\\')
        file_name=temp[len(temp)-1]
        print(file_name)
        
        #temp="C:/Users/Uday/Desktop/PyFit_Project/media/files"
        #file_name.replace('C:/Users/Uday/Desktop/PyFit_Project/media/files', '')
        #print(file_name)
        # global dfm,X_traing,X_testg,y_traing,y_testg
        # X_train,X_test,y_train,y_test=preprocessing(df)
        # dfm=df
        # X_traing=X_train
        # X_testg=X_test
        # y_traing=y_train
        # y_testg=y_test


        #print(X_train)
        
        # return render(request,'response.html',{'val':1,'df':df,'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test})
        return render(request,'response.html')
    else:
        return render(request,'home.html',{'name':'abcde'})

def response(request):
    return render(request,'response.html')

def abcd(request):
    #print(dfm)
    global file_name
    option=request.POST['option']
    print(option)
    option=int(option)
    if(option>=0 and option<=5):
        X_train,X_test,y_train,y_test=preprocessing(dfm)
        accuracy,precision,recall,f1,cm=mlmodel(option,X_train,X_test,y_train,y_test)
        return render(request,'temp.html',{'path':"media/models_created/"+file_name+".joblib",'option':option,'df':dfm,'range':range(len(dfm)),'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'cm':cm})
    
    if(option==6):
        X_train,X_test,y_train,y_test=preprocessing_lr(dfm)
        mse,rmse,mae,r2=linearRegression(option,X_train,X_test,y_train,y_test)
        return render(request,'temp.html',{'path':"media/models_created/"+file_name+".joblib",'option':option,'df':dfm,'mse':mse,'rmse':rmse,'mae':mae,'r2':r2})
    
    if(option==7):
        X_train,X_test,y_train,y_test=preprocessing_svr(dfm)
        mse,rmse,mae,r2=s_v_r(option,X_train,X_test,y_train,y_test)
        return render(request,'temp.html',{'path':"media/models_created/"+file_name+".joblib",'option':option,'df':dfm,'mse':mse,'rmse':rmse,'mae':mae,'r2':r2})





def index(request):
    #temp="44. Logistic Regression - Insurance Data (1).csv"
    temp="xyz"
    return render(request,'index.html',{'val':1,'file':temp})

def contact(request):
    return render(request,"aboutus.html")

def aboutpyfit(request):
    return render(request,"aboutpyfit.html")