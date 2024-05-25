# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:23:22 2023

@author: ghdtj
"""

import math as ma
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
def feature_1 (input_data):
    # 2번 가로축 Projection => 확률밀도함수로 변환 => 분산
    Column_Sum = sum(input_data)
    Total_Column_Sum = sum(Column_Sum)
    PDF = Column_Sum / Total_Column_Sum

    x = np.zeros(len(input_data))

    for i in range(len(input_data)):
        x[i] = i+1 
        
    E = sum(x*PDF)
    output_value = sum((x-E)*(x-E)*PDF)
    
    return output_value
###############################################################################

###############################################################################
def feature_2 (input_data):
    # 4번 세로축 Projection => 확률밀도함수로 변환 => 분산
    input_data = input_data.T
    Row_Sum = sum(input_data)
    Total_Row_Sum = sum(Row_Sum)
    PDF = Row_Sum / Total_Row_Sum

    x = np.zeros(len(input_data))

    for i in range(len(input_data)):
        x[i] = i+1 

    E = sum(x*PDF)
    output_value = sum((x-E)*(x-E)*PDF)
    
    return output_value
###############################################################################

###############################################################################
def feature_3 (input_data):
    # 6번 Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 분산
    input_dia_data = np.zeros(len(input_data))

    for i in range(len(input_data)):
        input_dia_data[i] = input_data[i,i]
    
    Total_dia_sum = sum(input_dia_data)
    PDF = input_dia_data / Total_dia_sum
    
    x = np.zeros(len(input_data))

    for i in range(len(input_data)):
        x[i] = i+1 
    
    E = sum(x*PDF)
    output_value = sum((x-E)*(x-E)*PDF)
    
    return output_value
###############################################################################

###############################################################################
def feature_4 (input_data):
    # 9번 Anti-Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 분산
    input_Anti_dia_data = np.zeros(len(input_data))

    for i in range(len(input_data)):
        input_Anti_dia_data[i] = input_data[i,len(input_data)-1-i]
        
    Total_anti_dia_sum = sum(input_Anti_dia_data)
    PDF = input_Anti_dia_data / Total_anti_dia_sum
    
    x = np.zeros(len(input_data))

    for i in range(len(input_data)):
        x[i] = i+1 
    
    E = sum(x*PDF)
    output_value = sum((x-E)*(x-E)*PDF)
    
    return output_value
###############################################################################

###############################################################################
def feature_5 (input_data):
    # 10번 Anti-Diagonal 원소 배열 추출  => 0의 개수
    input_Anti_dia_data = np.zeros(len(input_data))

    for i in range(len(input_data)):
        input_Anti_dia_data[i] = input_data[i,len(input_data)-1-i]
        
    count = 0
    for i in input_Anti_dia_data:
        if np.all(i) == 0:
            count += 1
    return count
###############################################################################

x_set = np.array([], dtype = 'float32')
x_set = np.resize(x_set, (0,5))

for j in range(3):
    for i in range(1, 501):
        temp_name = str(j) + '_' + str(i) + '.csv'
        temp_image = pd.read_csv(temp_name, header = None)
        temp_image = temp_image.to_numpy(dtype = 'float32')
        
        x0 = feature_1(temp_image)
        x1 = feature_2(temp_image)
        x2 = feature_3(temp_image)
        x3 = feature_4(temp_image)
        x4 = feature_5(temp_image)
        
        x_feature = np.array([x0,x1,x2,x3,x4], dtype = 'float32')
        x_feature = np.resize(x_feature, (1,5))
        x_set = np.concatenate( (x_set,x_feature), axis = 0)

# 입력층 데이터 set 만들기
x_0_set = x_set[0:500].T
x_1_set = x_set[500:1000].T
x_2_set = x_set[1000:1500].T
x_set = x_set.T
dummy_x_3_set = np.ones([1500])

Training_x_set = np.vstack([x_set,dummy_x_3_set])

# 출력층 데이터 set 만들기    
y_0_set = np.zeros(1500)
y_1_set = np.zeros(1500)
y_2_set = np.zeros(1500)

for i in range(500):
    y_0_set[i] = 1
    y_1_set[i+500] = 1
    y_2_set[i+1000] = 1

y_set = np.stack([y_0_set,y_1_set,y_2_set])

# 매개변수 불러오기
v_total = pd.read_csv('C:/Users/ghdtj/w_hidden.csv', header = None)
w_total = pd.read_csv('C:/Users/ghdtj/w_output.csv', header = None)

v_total = v_total.to_numpy(dtype = 'float32')
w_total = w_total.to_numpy(dtype = 'float32')

LR = 0.0001

# epoch 400일 때의 정확도와 MSE를 구하는 코드
epoch = [400]

# epoch 1~400까지 반복 횟수마다의 정확도와 MSE를 구하는 코드
# epoch = [0]*400
# for h in range(len(epoch)):
#     epoch[h] = h+1

b_dummy = np.ones(1)
y_hat_nq = np.zeros([1500,3])

v_diff = np.zeros([6,4])
w_diff = np.zeros([5,3])

total_mse = np.zeros(epoch[-1])
accuracy = np.zeros(epoch[-1])
mse = np.zeros(1500)

# 모델 훈련시키는 과정
for o in range(len(epoch)):
    for z in range(epoch[o]):
        for n in range(1500):
            alpha = np.dot(np.array(v_total).T,Training_x_set[:,n])
            f_a = 1 / (1 + np.exp(-alpha))
            b = np.hstack([f_a,b_dummy])
            
            beta = np.dot(np.array(w_total).T, b)
            y_hat = 1 / (1 + np.exp(-beta))
            error = y_hat - y_set[:,n]
        
            e = 2*error*y_hat*(1-y_hat)
            for l in range(4):
                for q in range(3):
                    w_diff[l][q] = e[q]*f_a[l]
            
            w_00 = e[0]*w_total[0,0] + e[1]*w_total[0,1] + e[2]*w_total[0,2]
            w_10 = e[0]*w_total[1,0] + e[1]*w_total[1,1] + e[2]*w_total[1,2]
            w_20 = e[0]*w_total[2,0] + e[1]*w_total[2,1] + e[2]*w_total[2,2]
            w_30 = e[0]*w_total[3,0] + e[1]*w_total[3,1] + e[2]*w_total[3,2]
            
            w_4 = np.hstack([w_00,w_10,w_20,w_30])
            
            for m in range(6):
                for l in range(4):
                    v_diff[m][l] = f_a[l]*(1-f_a[l])*Training_x_set[m,n]*w_4[l]
            
            w_total = w_total - LR*w_diff
            v_total = v_total - LR*v_diff
            
            compare_y_hat = np.zeros([3,1500])
            y_hat_nq[n] = y_hat
            mse[n] = sum((y_hat - y_set[:,n])*(y_hat - y_set[:,n]))
    
    # MSE 구하기
    total_mse[o] = np.mean(mse)
    
    # 정확도 구하기
    conformity = 0        
    for n in range(1500):    
        max_idx = np.argmax(y_hat_nq[n])
        compare_y_hat[max_idx,n] = 1
        
        if np.all(compare_y_hat[:,n] == y_set[:,n]):
            conformity = conformity + 1
        else:
            conformity = conformity
    accuracy[o] = (conformity/1500)*100