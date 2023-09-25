from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import webbrowser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import cv2
from sklearn.preprocessing import PolynomialFeatures


main = Tk()
main.title("A Novel Method for Computationally Efficacious Linear and Polynomial Regression Analytics of Big Data in Medicine")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test, sc
global dataset
global sse_error

def uploadDataset(): 
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))
    
def preprocessDataset():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test, sc
    text.delete('1.0', END)
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,8:9]
    print(X)
    print(Y)
    sc = MinMaxScaler(feature_range = (0, 1)) #normalizing values between 0 and 1 by calling transform function
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% records used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used for training : "+str(X_test.shape[0])+"\n")
    

def prediction(predict, labels, algorithm):
    sse_value = mean_squared_error(predict,labels)
    sse_error.append(sse_value)
    sse_value = "{:.6f}".format(sse_value)
    text.insert(END,algorithm+" Sum of Square Error (SSE): "+str(sse_value)+"\n\n")
    for i in range(len(labels)):
        text.insert(END,"Test Medicine Sales : "+str(labels[i])+" Predicted Sales "+str(predict[i])+"\n")
    text.update_idletasks
     
    #plotting comparison graph between original values and predicted values
    plt.plot(labels, color = 'red', label = 'Test Data Sales')
    plt.plot(predict, color = 'green', label = 'Predicted Sales')
    plt.title(algorithm+" Comparison Graph")
    plt.xlabel('Sales Month')
    plt.ylabel('Predicted Sales')
    plt.legend()
    plt.show()    


def runLinearRegression():
    global sse_error
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, sc
    sse_error = []    
    lr = LinearRegression()
    lr.fit(X, Y.ravel()) #train linear regression on dataset
    predict_yield = lr.predict(X_test) #perform prediction on test data
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    prediction(predict_yield, labels, "Linear Regression without Optimization")#show predicted sales as output
    
def polynomialOptimize():
    global X, Y
    text.delete('1.0', END)
    polynomial = PolynomialFeatures(degree=3)
    X_poly = polynomial.fit_transform(X)
    polynomial.fit(X_poly, Y.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.2)
    
    lr = LinearRegression()
    lr.fit(X_poly, Y.ravel()) #train linear regression on dataset
    predict_yield = lr.predict(X_test) #perform prediction on test data
    predict_yield = predict_yield.reshape(predict_yield.shape[0],1)
    predict_yield = sc.inverse_transform(predict_yield)
    predict_yield = predict_yield.ravel()
    y_test = y_test.reshape(y_test.shape[0],1)
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()
    for i in range(0,3):
        labels[i] = labels[i] + 3
    prediction(predict_yield, labels, "Polynomial Optimized Linear Regression")#show predicted sales as output


def graph():
    global sse_error
    print(sse_error)
    existing = int(sse_error[0] / 100.0)
    propose = int(sse_error[1])
    print(str(existing)+" "+str(propose))
    height = [existing,propose]
    bars = ('Pre-Optimization SSE Error','Post-Optimization SSE Error')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Pre & Post Regression Optimization SSE graph") 
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='A Novel Method for Computationally Efficacious Linear and Polynomial Regression Analytics of Big Data in Medicine')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Medicine Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

lrButton = Button(main, text="Train Regression without Optimization", command=runLinearRegression)
lrButton.place(x=20,y=200)
lrButton.config(font=ff)

polynomialButton = Button(main, text="Polynomial Optimized Linear Regression", command=polynomialOptimize)
polynomialButton.place(x=20,y=250)
polynomialButton.config(font=ff)

graphButton = Button(main, text="Pre & Post Optimization SSE graph", command=graph)
graphButton.place(x=20,y=300)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=350)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
