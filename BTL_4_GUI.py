import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tkinter import *
from tkinter import messagebox
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import StackingRegressor

data = pd.read_csv('D:/tlu-document/học máy/advertising.csv')
data=data.values

X = data[:, :3]
y = data[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_linear = LinearRegression()
model_linear.fit(X_train,y_train)
y_pred_linear = model_linear.predict(X_test)

model_lasso = Lasso(alpha=1000)
model_lasso.fit(X_train,y_train)
y_pred_lasso = model_lasso.predict(X_test)

model_neural = MLPRegressor(hidden_layer_sizes=(200,200),solver='lbfgs',max_iter=100,activation='identity')
model_neural.fit(X_train,y_train)
y_pred_neural = model_neural.predict(X_test)

stacking_estimators = [('lr',LinearRegression()),('lasso',Lasso(alpha=1000),('mlpr',MLPRegressor(hidden_layer_sizes=(200,200),solver='lbfgs',max_iter=100,activation='identity')))]
model_stacking = StackingRegressor(estimators=stacking_estimators)
model_stacking.fit(X_train,y_train)
y_pred_stacking = model_stacking.predict(X_test)

##################GUI##################
ui = Tk()
ui.title("Du doan doanh thu")
ui.geometry("1000x500")

label_tv=Label(ui,text="TV:").grid(row = 0,column=0,padx=5,pady=5)
tb_tv=Entry(ui)
tb_tv.grid(row=0,column=1,padx=5,pady=5)

label_radio=Label(ui,text="Radio:").grid(row = 1,column=0,padx=5,pady=5)
tb_radio=Entry(ui)
tb_radio.grid(row=1,column=1,padx=5,pady=5)

label_news=Label(ui,text="Newspaper:").grid(row = 2,column=0,padx=5,pady=5)
tb_news=Entry(ui)
tb_news.grid(row=2,column=1,padx=5,pady=5)

label_score_linear=Label(ui)
label_score_linear.grid(row=4,column=0,padx=20,pady=5)
label_score_linear.configure(text="Linear Regression: "+ "\n"
                             +'MAE: '+str(mae(y_test,y_pred_linear))+'\n'
                             +'MSE: '+str(mse(y_test,y_pred_linear))+'\n'
                             +'NSE: '+str((1-(np.sum((y_test-y_pred_linear)**2)/np.sum((y_test-np.mean(y_test))**2))))+'\n'
                             +'R2: '+str(r2_score(y_test,y_pred_linear))+'\n')

label_score_lasso=Label(ui)
label_score_lasso.grid(row=4,column=1,padx=20,pady=5)
label_score_lasso.configure(text="Lasso: "+ "\n"
                             +'MAE: '+str(mae(y_test,y_pred_lasso))+'\n'
                             +'MSE: '+str(mse(y_test,y_pred_lasso))+'\n'
                             +'NSE: '+str((1-(np.sum((y_test-y_pred_lasso)**2)/np.sum((y_test-np.mean(y_test))**2))))+'\n'
                             +'R2: '+str(r2_score(y_test,y_pred_lasso))+'\n')

label_score_neural=Label(ui)
label_score_neural.grid(row=4,column=2,padx=20,pady=5)
label_score_neural.configure(text="Neural Network: "+ "\n"
                             +'MAE: '+str(mae(y_test,y_pred_neural))+'\n'
                             +'MSE: '+str(mse(y_test,y_pred_neural))+'\n'
                             +'NSE: '+str((1-(np.sum((y_test-y_pred_neural)**2)/np.sum((y_test-np.mean(y_test))**2))))+'\n'
                             +'R2: '+str(r2_score(y_test,y_pred_neural))+'\n')

label_score_stacking=Label(ui)
label_score_stacking.grid(row=4,column=3,padx=20,pady=5)
label_score_stacking.configure(text="Stacking: "+ "\n"
                             +'MAE: '+str(mae(y_test,y_pred_stacking))+'\n'
                             +'MSE: '+str(mse(y_test,y_pred_stacking))+'\n'
                             +'NSE: '+str((1-(np.sum((y_test-y_pred_stacking)**2)/np.sum((y_test-np.mean(y_test))**2))))+'\n'
                             +'R2: '+str(r2_score(y_test,y_pred_stacking))+'\n')

test_linear=Label(ui,text="Gia tri du doan theo Linear Regression: ").grid(row=5,column=0,padx=5,pady=5)
test_linear_label=Label(ui,text=" ... ")
test_linear_label.grid(row=5,column=1,padx=5,pady=5)

test_lasso=Label(ui,text="Gia tri du doan theo Lasso: ").grid(row=6,column=0,padx=5,pady=5)
test_lasso_label=Label(ui,text=" ... ")
test_lasso_label.grid(row=6,column=1,padx=5,pady=5)

test_neural=Label(ui,text="Gia tri du doan theo Neural Network: ").grid(row=7,column=0,padx=5,pady=5)
test_neural_label=Label(ui,text=" ... ")
test_neural_label.grid(row=7,column=1,padx=5,pady=5)

test_stacking=Label(ui,text="Gia tri du doan theo Stacking: ").grid(row=8,column=0,padx=5,pady=5)
test_stacking_label=Label(ui,text=" ... ")
test_stacking_label.grid(row=8,column=1,padx=5,pady=5)

def dudoan_linear():
    tv=tb_tv.get()
    radio=tb_radio.get()
    news=tb_news.get()
    if((tv=='')or(radio=='')or(news=='')):
        messagebox.showinfo("Thong bao","Yeu cau nhap day du thong tin!")
    else:
        X_nhap=np.array([tv,radio,news],dtype=np.float64).reshape(1,-1)
        nhandudoan_linear=model_linear.predict(X_nhap)
        test_linear_label.configure(text=nhandudoan_linear)
        
def dudoan_lasso():
    tv=tb_tv.get()
    radio=tb_radio.get()
    news=tb_news.get()
    if((tv=='')or(radio=='')or(news=='')):
        messagebox.showinfo("Thong bao","Yeu cau nhap day du thong tin!")
    else:
        X_nhap=np.array([tv,radio,news],dtype=np.float64).reshape(1,-1)
        nhandudoan_lasso=model_lasso.predict(X_nhap)
        test_lasso_label.configure(text=nhandudoan_lasso)

def dudoan_neural():
    tv=tb_tv.get()
    radio=tb_radio.get()
    news=tb_news.get()
    if((tv=='')or(radio=='')or(news=='')):
        messagebox.showinfo("Thong bao","Yeu cau nhap day du thong tin!")
    else:
        X_nhap=np.array([tv,radio,news],dtype=np.float64).reshape(1,-1)
        nhandudoan_neural=model_neural.predict(X_nhap)
        test_neural_label.configure(text=nhandudoan_neural)

def dudoan_stacking():
    tv=tb_tv.get()
    radio=tb_radio.get()
    news=tb_news.get()
    if((tv=='')or(radio=='')or(news=='')):
        messagebox.showinfo("Thong bao","Yeu cau nhap day du thong tin!")
    else:
        X_nhap=np.array([tv,radio,news],dtype=np.float64).reshape(1,-1)
        nhandudoan_stacking=model_stacking.predict(X_nhap)
        test_stacking_label.configure(text=nhandudoan_stacking)


submit_linear=Button(ui,text="Du doan theo Linear",command=dudoan_linear).grid(row=9,column=0,padx=5,pady=20)
submit_linear=Button(ui,text="Du doan theo Lasso",command=dudoan_lasso).grid(row=9,column=1,padx=5,pady=20)
submit_linear=Button(ui,text="Du doan theo Neural Network",command=dudoan_neural).grid(row=9,column=2,padx=5,pady=20)
submit_linear=Button(ui,text="Du doan theo Stacking",command=dudoan_stacking).grid(row=9,column=3,padx=5,pady=20)
ui.mainloop()



