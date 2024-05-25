# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:24:20 2023

@author: ghdtj
"""

import math as ma
import numpy as np
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
