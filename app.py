from flask import Flask,request,render_template,url_for
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import joblib as joblib
import os
from vnstock import *
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint #lưu lại huấn luyện tốt nhất
from tensorflow.keras.models import load_model #tải mô hình

#các lớp để xây dựng mô hình
from keras.models import Sequential #đầu vào
from keras.layers import LSTM #học phụ thuộc
from keras.layers import Dropout #tránh học tủ
from keras.layers import Dense #đầu ra

#kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score #đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error #đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error 

app =Flask(__name__)

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER

TYPE_TICKET = {
    'VNM': 'stock',
    'VNINDEX': 'index',
    'VN30F1M': 'derivative',
    'TCB': 'stock'
}

#hàm format date
def format_date(date):
    date_format = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return date_format

#Hàm đọc dữ liệu
def read_data(tickets, start_date, end_date): 
    return stock_historical_data(symbol=tickets,
                            start_date=start_date,
                            end_date=end_date,
                            resolution='1D', type=TYPE_TICKET[tickets], source='DNSE')

#Hàm visualization
def visualization_data(df):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['time'], df['close'])
    ax.set_title('Close Price History')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price (VNĐ)', fontsize=18)
    fig.savefig('static/IMG/visualize_data.png', format='png')

#Hàm handle data
def handle_data(df):
    data = pd.DataFrame(df,columns=['time','close'])
    data.index = data.time
    data.drop('time',axis=1,inplace=True)
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8) #set length for data train - test

    # chuẩn hoá dữ liệu 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # lấy dữ liệu train
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train=[],[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0]) #lấy 60 giá đóng cửa liên tục
        y_train.append(train_data[i, 0]) #lấy ra giá đóng cửa ngày hôm sau

    #Chuyển x_train, y_train thành mảng
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #Reshape x_train, y_train thành mảng 1 chiều
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    y_train = np.reshape(y_train,(y_train.shape[0],1))

    # lấy dữ liệu test
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test, scaler, training_data_len, data

# Xây dựng mô hình
def build_model(x_train):
    model = Sequential()

    #2 lớp LSTM
    model.add(LSTM(units=128,input_shape=(x_train.shape[1],1),return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dropout(0.5)) #loại bỏ 1 số đơn vị tránh học tủ (overfitting)
    model.add(Dense(1)) #output đầu ra 1 chiều

    #Đo sai số bình phương trung bình có sử dụng trình tối ưu hóa adam
    model.compile(loss='mean_squared_error',optimizer='adam')

    return model

# Huấn luyện mô hình
def model_training(model, x_train, y_train):
    save_model = "save_model.keras"
    best_model = ModelCheckpoint(save_model,monitor='loss',verbose=2,save_best_only=True,mode='auto')
    model.fit(x_train,y_train,epochs=100,batch_size=50,verbose=2,callbacks=[best_model])

# Visualize predict data
def visualize_predict_data(train, valid):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('LSTM')
    ax.set_xlabel('Thời gian', fontsize=18)
    ax.set_ylabel('Giá đóng cửa (VNĐ)', fontsize=18)
    ax.plot(train['close'])
    ax.plot(valid[['close','Predictions']])
    ax.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    fig.savefig('static/IMG/visualize_predict_data.png', format='png')

# Dự giá cổ phiếu
def predict(tickets, date, scaler, final_model):
    end_date_history = format_date(date) - datetime.timedelta(days=1)
    start_date_history = format_date(str(end_date_history)) - datetime.timedelta(days=60)
    test_df = read_data(tickets, str(start_date_history), str(end_date_history))
    test_data = pd.DataFrame(test_df,columns=['time','close'])
    test_data.index = test_data.time
    test_data.drop('time',axis=1,inplace=True)
    data = test_data.values
    scaled_data = scaler.transform(data)
    X_test = []
    X_test.append(scaled_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = final_model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    return pred_price[0][0]

@app.route('/')
def index():
    # XÂY DỰNG MÔ HÌNH PREDICT
    # read data from vnstock
    # df = read_data(tickets, '2021-01-01', '2024-01-15')
    # handle data
    # x_train, y_train, x_test, y_test, scaler, training_data_len, data = handle_data(df)

    # xây dựng mô hình
    # model = build_model(x_train)

    #huấn luyện mô hình 
    # model_training(model, x_train, y_train)

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def home():
    if request.method =='POST':
        # request data from form
        tickets = request.form['tickets']
        date = request.form['date']
        
        # read data from vnstock
        df = read_data(tickets, '2021-01-01', '2024-01-15')

        # draw visulize for data frame
        visualization_data(df)
        image_visualize='visualize_data.png'
        image_visualize=os.path.join(app.config['UPLOAD_FOLDER'],image_visualize)

        # handle data
        x_train, y_train, x_test, y_test, scaler, training_data_len, data = handle_data(df)

        # xây dựng mô hình dự đoán giá cổ phiếu
        final_model = load_model("save_model.keras")
        predictions = final_model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # kiểm tra độ chính xác của mô hình
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        model_accuracy = [r2, mae, mape]

        # draw visualize predict
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        visualize_predict_data(train, valid)
        image_visualize_predict='visualize_predict_data.png'
        image_visualize_predict=os.path.join(app.config['UPLOAD_FOLDER'],image_visualize_predict)

        # DỰ ĐOÁN GIÁ VÀ SO SÁNH VỚI GIÁ CUỐI NGÀY
        pred_price = predict(tickets, date, scaler, final_model)
        
        actual_df = read_data(tickets, date, date)
        weekno = format_date(date).weekday()
        if weekno < 5:
            actual_price = actual_df['close'].values[0]
        else :
            actual_price = 'There are no closed prices on weekends'

    return render_template('predict.html',date=date, tickets=tickets, image_visualize=image_visualize, model_accuracy=model_accuracy, image_visualize_predict=image_visualize_predict, pred_price=pred_price, actual_price=actual_price)

if __name__ == '__main__':
    app.run(debug=True)