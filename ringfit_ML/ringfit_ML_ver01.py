#!/usr/bin/env python
# coding: utf-8

# # 준비

# ## 모듈

# In[1]:


import os
import shutil
import json
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema


# ## 태그

# In[2]:


body_part = ["Nose_x", "Nose_y",
             "Neck_x", "Neck_y",
             "RShoulder_x", "RShoulder_y",
             "RElbow_x", "RElbow_y",
             "RWrist_x", "RWrist_y",
             "LShoulder_x", "LShoulder_y",
             "LElbow_x", "LElbow_y",
             "LWrist_x", "LWrist_y",
             "MidHip_x", "MidHip_y",
             "RHip_x", "RHip_y",
             "RKnee_x", "RKnee_y",
             "RAnkle_x", "RAnkle_y",
             "LHip_x", "LHip_y",
             "LKnee_x", "LKnee_y",
             "LAnkle_x", "LAnkle_y",
             "REye_x", "REye_y",
             "LEye_x", "LEye_y",
             "REar_x", "REar_y",
             "LEar_x", "LEar_y",
             "LBigToe_x", "LBigToe_y",
             "LSmallToe_x", "LSmallToe_y",
             "LHeel_x", "LHeel_y",
             "RBigToe_x", "RBigToe_y",
             "RSmallToe_x", "RSmallToe_y",
             "RHeel_x", "RHeel_y"]

main_body_part = ["Neck_x", "Neck_y", 
     "RShoulder_x", "RShoulder_y", 
     "RElbow_x", "RElbow_y", 
     "RWrist_x", "RWrist_y", 
     "LShoulder_x", "LShoulder_y", 
     "LElbow_x", "LElbow_y", 
     "LWrist_x", "LWrist_y", 
     "MidHip_x", "MidHip_y", 
     "RHip_x", "RHip_y", 
     "RKnee_x", "RKnee_y", 
     "RAnkle_x", "RAnkle_y", 
     "LHip_x", "LHip_y", 
     "LKnee_x", "LKnee_y", 
     "LAnkle_x", "LAnkle_y"]

body_part_with_confidence = ["Nose_x", "Nose_y", "Nose_cfd",
                             "Neck_x", "Neck_y", "Neck_cfd", 
                             "RShoulder_x", "RShoulder_y", "Rshoulder_cfd", 
                             "RElbow_x", "RElbow_y", "RElbow_cfd", 
                             "RWrist_x", "RWrist_y", "RWrist_cfd",
                             "LShoulder_x", "LShoulder_y", "LShoulder_cfd",
                             "LElbow_x", "LElbow_y", "LElbow_cfd",
                             "LWrist_x", "LWrist_y", "LWrist_cfd",
                             "MidHip_x", "MidHip_y", "MidHip_cfd",
                             "RHip_x", "RHip_y", "RHip_cfd",
                             "RKnee_x", "RKnee_y", "RKnee_cfd",
                             "RAnkle_x", "RAnkle_y", "RAnkle_cfd",
                             "LHip_x", "LHip_y", "LHip_cfd",
                             "LKnee_x", "LKnee_y", "LKnee_cfd",
                             "LAnkle_x", "LAnkle_y", "LAnkle_cfd",
                             "REye_x", "REye_y", "REye_cfd",
                             "LEye_x", "LEye_y", "LEye_cfd",
                             "REar_x", "REar_y", "REar_cfd",
                             "LEar_x", "LEar_y", "LEar_cfd",
                             "LBigToe_x", "LBigToe_y", "LBigToe_cfd",
                             "LSmallToe_x", "LSmallToe_y", "LSmallToe_cfd",
                             "LHeel_x", "LHeel_y", "LHeel_cfd",
                             "RBigToe_x", "RBigToe_y", "RBigToe_cfd",
                             "RSmallToe_x", "RSmallToe_y", "RSmallToe_cfd",
                             "RHeel_x", "RHeel_y", "RHeel_cfd"]

BackAndFace = [[8,1],[1,0],[0,15],[0,16],[16,18]]
RightArm = [[1,2],[2,3],[3,4]]
LeftArm = [[1,5],[5,6],[6,7]]
RightLeg = [[8,9],[9,10],[10,11],[11,22],[11,23],[11,24]]
LeftLeg = [[8,12],[12,13],[13,14],[14,19],[14,20],[14,21]]

BackAndFace_pt = [0,1,8,15,16,18]
RightArm_pt = [2,3,4]
LeftArm_pt = [5,6,7]
RightLeg_pt = [9,10,11,22,23,24]
LeftLeg_pt = [12,13,14,19,20,21]


# ## Openpose가 생성한 .json 파일 정리
# * 생성 후 최초 1회만 실행
# * 다시 실행하지 말 것

# In[3]:


# src = "./new_data/"


# In[4]:


# # 이름이 너무 길어서 줄였음(001_010.json)
# filelist = os.listdir(src)
# for file in filelist:
#     new_name = file[5:8] + '_' + file[18:21] + file[-5:]
#     os.rename(src+file, src+new_name)


# In[5]:


# filelist = os.listdir(src)
# for file in filelist:
#     new_src = src+file[:3]
#     if not os.path.isdir(new_src):
#         os.mkdir(new_src)
#     shutil.move(src+file, src+file[:3]+'/'+file)


# ### 생성데이터 아웃풋

# In[6]:


# src = "output_001_060/"


# In[7]:


# # 이름이 너무 길어서 줄였음(001_010.json)
# filelist = os.listdir(src)
# for file in filelist:
#     new_name = file[6:9] + '_' + file[19:22] + file[-5:]
#     os.rename(src+file, src+new_name)


# In[8]:


# filelist = os.listdir(src)
# for file in filelist:
#     new_src = src+file[:3]
#     if not os.path.isdir(new_src):
#         os.mkdir(new_src)
#     shutil.move(src+file, src+file[:3]+'/'+file)


# ## 기준 운동 설정

# In[9]:


std_path = 'train_data_240_json/001/'


# ## 함수

# ### 거리 계산

# In[10]:


# 두 점 사이의 거리 계산
# 단위: 픽셀
def cal_distance(point_a, point_b):
    dis = np.array(point_a) - np.array(point_b)
    dist = np.linalg.norm(dis)
    return dist


# ### 각도 계산

# In[11]:


# a, b, c 순서대로 높은 좌표부터 넣어줘야 함
# point_b에 반드시 세 좌표 중 가운데 좌표를 넣어줘야함
# ex) point_a = 엉덩이, point_b = 무릎, point_c = 발목
# create vectors, 두 점 사이 거리 계산
def cal_degree(point_a, point_b, point_c):
    dis_ba = np.array(point_a) - np.array(point_b)
    dis_bc = np.array(point_c) - np.array(point_b)

    # calculate angle
    cosine_angle = np.dot(dis_ba, dis_bc) / (np.linalg.norm(dis_ba) * np.linalg.norm(dis_bc))

    angle = np.arccos(cosine_angle)
    inner_angle = round(np.degrees(angle), 3)
    if point_c[1] < point_a[1]:
        inner_angle = -inner_angle
    
    return inner_angle


# ### JSON -> DataFrame 변환

# In[12]:


# path는 ".../"으로 끝나야 함
# savgol_filter를 사용해 smoothing
def json_to_df(path):
    json_list = list()
    for fname in os.listdir(path):
        with open(path+fname, "r") as json_file:
            js = json.load(json_file, encoding="utf-8")
            if not js['people']:
                continue
            json_list.append(js["people"][0]["pose_keypoints_2d"]) 
            #관절의 태그 정보를 가지고있는 좌표

    df_data = list()
    for item in json_list:
        temp = list()
        for idx, d in enumerate(item):
            temp.append(d)
        df_data.append(temp)

    df = pd.DataFrame(df_data, columns=body_part_with_confidence)
    df.replace(0, np.nan, inplace=True)
    df = df.interpolate()
    
    df["REar_x"] = 0.0
    df["REar_y"] = 0.0
    df["REar_cfd"] = 0.0
    df['Angle_Rleg'] = df.apply(lambda x:cal_degree([x.RHip_x, x.RHip_y], [x.RKnee_x, x.RKnee_y], [x.RAnkle_x, x.RAnkle_y]), axis=1)
    df['Angle_Lleg'] = df.apply(lambda x:cal_degree([x.LHip_x, x.LHip_y], [x.LKnee_x, x.LKnee_y], [x.LAnkle_x, x.LAnkle_y]), axis=1)
    df['Angle_waist'] = df.apply(lambda x:cal_degree([x.Neck_x, x.Neck_y], [x.MidHip_x, x.MidHip_y], [0, x.RHip_y]), axis=1)

    mod_df = savgol_filter(df, 51, 7, axis = 0)
    mod_df = pd.DataFrame(mod_df, columns=df.columns)
    
    return mod_df


# ### Cycle 찾기

# In[13]:


# Angle_Rleg를 사용해 cycle 탐색
def get_cycle(path):
    mod_df = json_to_df(path) # mod_df는 smoothing된 DataFrame
    
    x = [i for i in range(len(mod_df))]
    y = mod_df["Angle_Rleg"].to_numpy()
    max_idx = argrelextrema(y, np.greater)
    #local maximum의 인덱스 찾음
    min_idx = argrelextrema(y, np.less)
    #local minimum의 인덱스 찾음
    order1 = np.gradient(y) #1차미분
    order2 = np.gradient(order1)#2차미분
    
    new_max = []
    for idx in max_idx[0]:
        if order1[idx-1] > 0 and order1[idx+1] < 0 and round(order2[idx],3) < 0:
            new_max.append(idx)

    new_min = []
    for idx in min_idx[0]:
        if order1[idx-1] < 0 and order1[idx+1] > 0 and round(order2[idx],3) > 0:
            new_min.append(idx)
    
    for i in range(len(new_min)-1):
        if y[new_min[i+1]] < y[new_min[i]] * 0.8:
            idx = new_min[i+1]
            break
            
    for i in range(len(new_max)):
        if idx < new_max[i]:
            start = new_max[i-1]
            end = new_max[i]
            break
            
    x_new = x[start:end+1]
    y_new = y[start:end+1]
    return idx, x_new, y_new # idx가 운동 중간 지점


# ### Similarity

# In[14]:


def cal_similarity(p, q):
    length = min(len(p), len(q))
    common = []
    for i in range(length):
        common.append(min(p[i], q[i]))
    return sum(common)*100/np.sum(q)


# ### 기준 영상 설정

# In[15]:


def set_standard(path):
    mid, cycle_std, cycle_value = get_cycle(path)

    # polynomial regression
    mod_df_ori = json_to_df(path)
    df = mod_df_ori.iloc[cycle_std[0]:cycle_std[-1]+1, :]
    x = [(num-cycle_std[0])/(len(cycle_std)-1) for num in cycle_std]
    X = np.array(x)[:, np.newaxis]
    X_fit = np.arange(0, 1, 0.01)[:, np.newaxis]
    
    # sim_lleg
    y = [180-value for value in list(df.Angle_Lleg)] #취존
    y = np.array(y)

    lr = LinearRegression()
    model = PolynomialFeatures(degree = 20)
    X_new = model.fit_transform(X)
    lr.fit(X_new, y)
    y_std_fit = lr.predict(model.fit_transform(X_fit))

    # sim_rleg
    y2 = [180-value for value in list(df.Angle_Rleg)]
    y2 = np.array(y2)

    lr2 = LinearRegression()
    model2 = PolynomialFeatures(degree = 20)
    X_new_2 = model2.fit_transform(X)
    lr2.fit(X_new, y2)
    y_std_fit_2 = lr2.predict(model2.fit_transform(X_fit))
    return y_std_fit, y_std_fit_2


# ### 기준 영상과 비교

# In[16]:


def compare(path, gender, save = False):
    global std_path, std_path_fe
    if gender == 1:
        y_std, y_std_2 = set_standard(std_path)
    elif gender == 2:
        y_std, y_std_2 = set_standard(std_path_fe)
    else:
        return "Unavailable gender"
    
    mid, cycle_cpr, cycle_cpr_value = get_cycle(path)

    # polynomial regression
    mod_df_ori = json_to_df(path)
    df = mod_df_ori.iloc[cycle_cpr[0]:cycle_cpr[-1]+1, :]
    x = [(num-cycle_cpr[0])/(len(cycle_cpr)-1) for num in cycle_cpr]
    X = np.array(x)[:, np.newaxis]
    X_fit = np.arange(0, 1, 0.01)[:, np.newaxis]
    
    # sim_lleg
    y = [180-value for value in list(df.Angle_Lleg)]
    y = np.array(y)

    lr = LinearRegression()
    model = PolynomialFeatures(degree = 20)
    X_new = model.fit_transform(X)
    lr.fit(X_new, y)
    y_new_fit = lr.predict(model.fit_transform(X_fit))

    p = y_std
    standard_1 = np.where(p < 0, -p, p)
    q = y_new_fit
    compare_1 = np.where(q < 0, -q, q)

    # sim_rleg
    y2 = [180-value for value in list(df.Angle_Rleg)]
    y2 = np.array(y2)

    lr2 = LinearRegression()
    model2 = PolynomialFeatures(degree = 20)
    X_new_2 = model2.fit_transform(X)
    lr2.fit(X_new_2, y2)
    y_new_fit_2 = lr2.predict(model2.fit_transform(X_fit))

    p2 = y_std_2
    standard_2 = np.where(p2 < 0, -p2, p2)
    q2 = y_new_fit_2
    compare_2 = np.where(q2 < 0, -q2, q2)
    
    # similarity 그래프 저장 용도
    if save == True:
        standard_0 = (standard_1 + standard_2)/2
        compare_0 = (compare_1 + compare_2)/2
        plt.plot(X_fit, standard_0, c='red')
        plt.plot(X_fit, compare_0, c='blue')
        plt.title('Similarity Graph')
        # 저장 파일명 설정
        plt.savefig(f'sim_graph_{path[-4:-1]}.jpg')
        plt.close()

    # save result
    min_angle_lleg = min(df.Angle_Lleg)
    min_angle_rleg = min(df.Angle_Rleg)
    min_angle_waist = min(df.Angle_waist)
    down_time = mid - cycle_cpr[0]
    up_time = cycle_cpr[-1] - mid
    sim_lleg = cal_similarity(compare_1, standard_1)
#     if sim_lleg < 0:
#         sim_lleg = -sim_lleg
    sim_rleg = cal_similarity(compare_2, standard_2)
#     if sim_rleg < 0:
#         sim_rleg = -sim_rleg
    similarity = (sim_lleg + sim_rleg)/2

    result = {'min_angle_lleg': min_angle_lleg, 'min_angle_rleg': min_angle_rleg, 'min_angle_waist': min_angle_waist,             'down_time': down_time, 'up_time': up_time, 'similarity': similarity}
    return result


# ### 스쿼트 판단 & 피드백

# In[17]:


def squat(dict_, gender):
    global male_standard, female_standard
    if gender == 1:
        standard = male_standard
    elif gender == 2:
        standard = female_standard
    else:
        return "Unavailable gender"
    
    advice = {}
#     append = True
    
    # 무릎 각도 차이
    lleg_mean = (np.sum(standard.min_angle_lleg) - min(standard.min_angle_lleg) - max(standard.min_angle_lleg))/    (len(standard)-2)
    diff_lleg = lleg_mean - dict_['min_angle_lleg']
    rleg_mean = (np.sum(standard.min_angle_rleg) - min(standard.min_angle_rleg) - max(standard.min_angle_rleg))/    (len(standard)-2)
    diff_rleg = rleg_mean - dict_['min_angle_rleg']
    diff_leg = (diff_lleg + diff_rleg)/2
#     if diff_leg >= 0:
#         advice[1] = f"평균보다 무릎이 약 {diff_leg:2.2f}도 더 구부러졌습니다."
#     else:
#         advice[1] = f"평균보다 무릎이 약 {-diff_leg:2.2f}도 덜 구부러졌습니다."
    advice[1] = f"{diff_leg:3.3f}"
#     if (lleg_mean + rleg_mean)*0.95/2 <= (dict_['min_angle_lleg']+dict_['min_angle_rleg'])/2 <= (lleg_mean + rleg_mean)*1.05/2:
#         pass
#     else:
#         append = False
        
    
    # 허리 각도 차이
    waist_mean = (np.sum(standard.min_angle_waist) - min(standard.min_angle_waist) - max(standard.min_angle_waist))    /(len(standard)-2)
    diff_waist = waist_mean - dict_['min_angle_waist']
#     if diff_waist >= 0:
#         advice[2] = f"평균보다 허리가 약 {diff_waist:2.2f}도 더 굽혀졌습니다."
#     else:
#         advice[2] = f"평균보다 허리가 약 {-diff_waist:2.2f}도 덜 굽혀졌습니다."
    advice[2] = f"{diff_waist:3.3f}"
#     if waist_mean*0.95 <= dict_['min_angle_waist'] <= waist_mean*1.05:
#         pass
#     else:
#         append = False
    
        
    # 운동 시간 차이
    # down
    dtime_mean = (np.sum(standard.down_time) - min(standard.down_time) - max(standard.down_time))    /(len(standard)-2)
    diff_dtime_frame = dtime_mean - dict_['down_time']
    diff_dtime = diff_dtime_frame/30
#     if diff_dtime >= 0:
#         advice[3] = f"평균보다 약 {diff_dtime:2.2f}초 빨리 앉았습니다."
#     else:
#         advice[3] = f"평균보다 약 {-diff_dtime:2.2f}초 느리게 앉았습니다."
    advice[3] = f"{diff_dtime:3.3f}"
#     if dtime_mean*0.95 <= dict_['down-time'] <= dtime_mean*1.05:
#         pass
#     else:
#         append = False
    #up
    utime_mean = (np.sum(standard.up_time) - min(standard.up_time) - max(standard.up_time))    /(len(standard)-2)
    diff_utime_frame = utime_mean - dict_['up_time']
    diff_utime = diff_utime_frame/30
#     if diff_utime >= 0:
#         advice[4] = f"평균보다 약 {diff_utime:2.2f}초 빨리 일어났습니다."
#     else:
#         advice[4] = f"평균보다 약 {-diff_utime:2.2f}초 느리게 일어났습니다."
    advice[4] = f"{diff_utime:3.3f}"
#     if utime_mean*0.95 <= dict_['up_time'] <= utime_mean*1.05:
#         pass
#     else:
#         append = False
        
    # 유사도
    sim_leg = dict_['similarity']
#     advice[5] = "운동의 전체적인 유사도는 {:2.2f}%입니다.".format(100 - sim_leg)
    advice[5] = f"{sim_leg:3.3f}"
#     if sim_leg >= 95:
#         pass
#     else:
#         append = False

#     if append:
#         male_standard.loc[-1] = dict_
    return advice


# ### Cycle 통일

# In[18]:


def synchronize(com_path):
    global std_path
    std_df = json_to_df(std_path)
    com_df = json_to_df(com_path)
    std_mid, std_cycle, std_cycle_y = get_cycle(std_path)
    com_mid, com_cycle, com_cycle_y = get_cycle(com_path)
    
    # 전체 데이터프레임에서 한 사이클에 해당하는 부분만 분리
    std_down_df = std_df.iloc[min(std_cycle):std_mid+1, :]
    std_up_df = std_df.iloc[std_mid+1:max(std_cycle)+1, :]
    
    com_cycle_df = com_df.iloc[min(com_cycle):max(com_cycle)+1, :]
    com_down_df = com_df.iloc[min(com_cycle):com_mid+1, :]
    com_up_df = com_df.iloc[com_mid+1:max(com_cycle)+1, :]

    if len(com_down_df) == len(std_down_df):
        pass
    elif len(com_down_df) > len(std_down_df):
        lack = len(com_down_df) - len(std_down_df)
        append_idx = []
        for m in range(lack):
            append_idx.append((m+1)*(len(std_down_df)//(lack+1)))
        for num in append_idx:
            std_down_df.loc[min(std_cycle)+num+0.5] = None
        std_down_df.sort_index(inplace = True)
        std_down_df.interpolate(method = 'linear', inplace = True)
    else:
        left = len(std_down_df) - len(com_down_df)
        drop_idx = []
        for m in range(left):
            drop_idx.append(m*(len(std_down_df)//left)+(len(std_down_df)//left)//2)
        drop_idx.reverse() #뒤에 있는 인덱스부터 빼줘야 쉽게 빠짐
        for num in drop_idx:
            std_down_df.drop(min(std_cycle)+num, inplace = True)

    if len(com_up_df) == len(std_up_df):
        pass
    elif len(com_up_df) > len(std_up_df):
        lack = len(com_up_df) - len(std_up_df)
        append_idx = []
        for m in range(lack):
            append_idx.append((m+1)*(len(std_up_df)//(lack+1)))
        for num in append_idx:
            std_up_df.loc[std_mid+num+0.5] = None
        std_up_df.sort_index(inplace = True)
        std_up_df.interpolate(method = 'linear', inplace = True)
    else:
        left = len(std_up_df) - len(com_up_df)
        drop_idx = []
        for m in range(left):
            drop_idx.append(m*(len(std_up_df)//left)+(len(std_up_df)//left)//2)
        drop_idx.reverse() #뒤에 있는 인덱스부터 빼줘야 쉽게 빠짐
        for num in drop_idx:
            std_up_df.drop(std_mid+num, inplace = True)

    std_cycle_df = pd.concat([std_down_df, std_up_df])
    std_cycle_df.sort_index(inplace = True)
    
    std_start = std_cycle_df.iloc[0]
    com_start = com_cycle_df.iloc[0]

    # 기준 운동의 관절들을 비교 대상과 동일한 위치로 가져옴
    for part in body_part:
        diff = std_start[part] - com_start[part]
        std_cycle_df[part] -= diff
    std_cycle_df.reset_index(drop = True, inplace = True)
    com_cycle_df.reset_index(drop = True, inplace = True)
    return std_cycle_df, com_cycle_df


# ### 태그마다 기준과 일치하는 프레임 비율 계산

# In[19]:


def cal_ratio(com_path):
    global std_path, body_part, main_body_part
    std_cycle_df, com_cycle_df = synchronize(com_path)

    length = min(len(std_cycle_df), len(com_cycle_df))
    # 기준 범위에서 벗어난 관절 개수 출력
    out_info = [] # 범위를 벗어난 프레임 & 관절들 저장
    for idx in range(length):
        bp_outofrange = [] # 정해진 범위는 벗어난 관절들 모음
        for j in list(range(len(main_body_part)//2)):
            if cal_distance((std_cycle_df[main_body_part[2*j]][idx], std_cycle_df[main_body_part[2*j+1]][idx]),                           (com_cycle_df[main_body_part[2*j]][idx], com_cycle_df[main_body_part[2*j+1]][idx])) <= 20:
                pass
            else:
                bp_outofrange.append(j+1)
        out_info.append((idx, bp_outofrange))

    frame_ratio = pd.DataFrame(columns = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',                                     'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'])
    for num in range(1, 15):
        frame_list = []
        for part in out_info:
            if num in part[1]:
                frame_list.append(out_info[0])
        ratio = 1 - len(frame_list)/len(out_info)
        frame_ratio.loc[com_path[-4:-1], frame_ratio.columns[num-1]] = ratio
        # row index(com_path[-4:-1]) 수정할 필요 있으면 할 것
    return out_info, frame_ratio


# ### 시각화
# * Time Graph & Similarity Graph
# * 사용자 영상 배경 + 기준&사용자 스켈레톤
# * 내려올 때&올라올 때 프레임 수 일치시킴
# * 기준은 파란색, 기준 범위에서 벗어난 좌표는 빨간색으로 표시

# In[20]:


def save_pic(com_path):
    global std_path
    std_mid, std_cycle, std_cycle_y = get_cycle(std_path)
    std_x = [el - std_cycle[0] for el in std_cycle]

    com_mid, com_cycle, com_cycle_y = get_cycle(com_path)
    com_x = [el - com_cycle[0] for el in com_cycle]

    plt.plot(std_x, std_cycle_y, c='red')
    plt.plot(com_x, com_cycle_y, c='blue')
    plt.title('Time Graph')
    # 저장 파일명 설정
    plt.savefig(f'time_graph_{com_path[-4:-1]}.jpg')
    plt.close()

    result = compare(com_path, 1, True)


# In[21]:


def visual_comparing(com_path):
    global std_path
    std_mod_df, com_mod_df = synchronize(com_path)
    out_info, frame_ratio = cal_ratio(com_path)

    draw_mod_df = std_mod_df.drop(columns=["Angle_Rleg", "Angle_Lleg", "Angle_waist"])
    mod_df_list = draw_mod_df.values.tolist()

    draw_mod_df2 = com_mod_df.drop(columns=["Angle_Rleg", "Angle_Lleg", "Angle_waist"])
    mod_df_list2 = draw_mod_df2.values.tolist()

    # 사용자 운동 영상의 경로
#    workout = cv2.VideoCapture(f"/home/centos/ringfit_web/upload/user_000.mp4")	
    workout = cv2.VideoCapture(f"train_video/train_{com_path[-4:-1]}.mp4")
#     workout = cv2.VideoCapture(f"new_video/test_{com_path[-4:-1]}.mp4")
#     workout = cv2.VideoCapture("D:/링피트/LogiCapture/2020-08-20_14-42-58_90000000000.mp4")
    mid,cycle_x_list,cycle_y_list = get_cycle(com_path)
    if workout.isOpened():
#         print('width: %d, height: %d' % (workout.get(3), workout.get(4))) # 영상의 너비와 높이 출력

        fourcc = cv2.VideoWriter_fourcc(*'VP90') # 코덱 정의
        # 여기도 저장 파일명 확인
        out30 = cv2.VideoWriter(f'result_{com_path[-4:-1]}_fps30.webm', fourcc, 30.0, (1280, 720)) # VideoWriter 객체 정의
      #  out15 = cv2.VideoWriter(f'result_{com_path[-4:-1]}_fps15.mp4', 0x00000021, 15, (1280, 720)) # VideoWriter 객체 정의
      #  out06 = cv2.VideoWriter(f'result_{com_path[-4:-1]}_fps6.mp4', 0x00000021, 6, (1280, 720)) # VideoWriter 객체 정의

        i = -cycle_x_list[0] # frame number
        while True:
            ret, img = workout.read()
            i += 1
            if i <= 0:
                continue

            # 색상표
            color1_list = [] # 몸통
            for temp in range(len(BackAndFace)):
                color1 = tuple([100+temp*25]*3)
                color1_list.append(color1)

            color2_list = [] # 팔
            for temp in range(len(RightArm)):
                color2 = tuple([100+temp*25]*3)
                color2_list.append(color1)

            color3_list = [] # 다리
            for temp in range(len(RightLeg)):
                color3 = tuple([100+temp*25]*3)
                color3_list.append(color1)

            try:
            # draw circle
                for j in range(len(body_part_with_confidence)//3): # 관절 갯수 25개
                    x = int(mod_df_list[i][3*j]) # x좌표
                    x2 = int(mod_df_list2[i][3*j]) # x좌표
                    y = int(mod_df_list[i][3*j+1]) # y좌표
                    y2 = int(mod_df_list2[i][3*j+1]) # y좌표
                    cfd = mod_df_list[i][3*j+2] # confidence score
                    cfd2 = mod_df_list2[i][3*j+2] # confidence score
                    if int(x) == 0 or int(y) == 0 or cfd <= 0.3:
                        continue # 좌표값이 0이거나 그 좌표값에 대한 confidence score가 0인 경우 skip(continue)
                    if int(x2) == 0 or int(y2) == 0 or cfd2 <= 0.3:
                        continue
                    if j in BackAndFace_pt:
                        color = color1_list[-1]
                    elif j in RightArm_pt + LeftArm_pt:
                        color = color2_list[-1]
                    elif j in RightLeg_pt + LeftLeg_pt:
                        color = color3_list[-1]
                    cv2.circle(img, (x,y), int(cfd*10), color,-1) # 색상표 만들어서 얼굴/몸통/팔/다리 색깔구분하기
                    cv2.circle(img, (x2,y2), int(cfd2*10), color, -1)

                # 한방에!

                Body_25 = [BackAndFace, RightArm, LeftArm, RightLeg, LeftLeg]
                for a, body in enumerate(Body_25): 
                    idx = 0
                    for pt1, pt2 in body:
                        start = (int(mod_df_list[i][3*pt1]), int(mod_df_list[i][3*pt1+1]))
                        end = (int(mod_df_list[i][3*pt2]), int(mod_df_list[i][3*pt2+1]))

                        start2 = (int(mod_df_list2[i][3*pt1]), int(mod_df_list2[i][3*pt1+1]))
                        end2 = (int(mod_df_list2[i][3*pt2]), int(mod_df_list2[i][3*pt2+1]))

                        if 0 in (start[0], start[1], end[0], end[1], start2[0], start2[1], end2[0], end2[1]):
                            continue
                        thickness = int((mod_df_list[i][3*pt1+2] + mod_df_list[i][3*pt2+2])*5)
                        thickness2 = int((mod_df_list2[i][3*pt1+2] + mod_df_list2[i][3*pt2+2])*5)

                        if a == 0:
                            color = color1_list[idx]
                            color2 = color1_list[idx]
                        elif a == 1 or a == 2:
                            color = color2_list[idx]
                            color2 = color2_list[idx]
                        elif a == 3 or a == 4:
                            color = color3_list[idx]
                            color2 = color3_list[idx]
                            if (pt1 in out_info[i][1]) or (pt2 in out_info[i][1]):
                                color = (0, 0, 255) # 기준은 빨간색
                                color2 = (255, 0, 0) # 사용자 영상은 파란색
                        cv2.line(img, start, end, color, thickness)
                        cv2.line(img, start2, end2, color2, thickness2)
                        idx += 1

            except:
                break

            out30.write(img) # fps30
          #  out15.write(img) # fps15
          #  out06.write(img) # fps6

            if ret:
                cv2.imshow('Visual Comparing', img)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
            else:
                break
    else:
        print("No Video")
    workout.release()
    cv2.destroyAllWindows()
    
    save_pic(com_path)


# ### 이 모든 걸 한방에

# In[22]:


def ringfit(com_path, gender):
    global male_train, female_train
    test = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',         'down_time', 'up_time', 'similarity'])
    result = compare(com_path, gender)
    output = squat(result, gender)
    test.loc[com_path[-4:-1]] = result

    if gender == 1:
        train_df = male_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
        train_x = train_df.iloc[:, :-1]
        train_y = train_df.iloc[:, -1]

    elif gender == 2:
        train_df = female_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
        train_x = female_train.iloc[:, :-1]
        train_y = female_train.iloc[:, -1]
    else:
        print("Unavailable Gender")

    out_info, ratio_df = cal_ratio(com_path)
    test = pd.concat([test, ratio_df], axis = 1)
    test['grade'] = None
    # 사용자 영상도 전체 운동 데이터프레임에 추가
    #     if gender == 1:
    #         new_train = pd.concat([male_train, test], axis = 0)
    #     elif gender == 2:
    #         new_train = pd.concat([female_train, test], axis = 0)
    test.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle', 'grade'], axis = 1, inplace = True)

    ## 여기부터 머신러닝
    # import한 모델로 머신러닝 모델 만들어서 돌리면 됨
    predict_result = []
    for _ in range(10):
        mlp = MLPClassifier(solver = 'lbfgs', max_iter = 500, early_stopping = True)
        mlp.fit(train_x, train_y)
        test_x = test 
        predict_result.append(mlp.predict(test_x)[0])
#         print(mlp.predict(test_x))
#         print(mlp.predict_proba(test_x))
    result_idx, count = 0, 0
    for idx, el in enumerate(['A', 'B', 'C', 'F']):
        if count < predict_result.count(el):
            count = predict_result.count(el)
            result_idx = idx
    final_grade = ['A', 'B', 'C', 'F'][result_idx]
    output['grade'] = final_grade
    # 사용자 영상의 grade 반영
#     new_train.iloc[-1, -1] = final_grade
#     if gender == 1:
#         new_train.to_csv("male_train_with_frame.csv", index = False)
#     elif gender == 2:
#         new_train.to_csv("female_train_with_frame.csv", index = False)
    
    with open(f'ringfit_output_{com_path[-4:-1]}.json', 'w') as ringfit:
        json.dump(output, ringfit)
    
    visual_comparing(com_path)
    
    return


# ## DataFrame 생성
# * csv 가지고 있으면 안 돌려도 되는 부분

# In[23]:


# # # 남자만 있으니까 male만 만들게요
# # # 이제 돌렸으니까 필요 없음
# # std_path = 'train_data_240_json/001/'
# label_data = pd.read_csv("new_data_labeling_2.csv", header = None)
# male_idx = []
# male_test_idx = []
# # female_idx = []
# for idx in range(len(label_data)):
#     if label_data[1][idx] in ('훈석', '정우'):
#         male_idx.append(idx+1)
#     else:
#         male_test_idx.append(idx+1)
# #     elif label_data[1][idx] in ('누군가'):
# #         female_idx.append(idx)
# print(male_idx)
# # print(female_idx)
# print(male_test_idx)

# # # 주석 처리한 부분 -> 새로운 생성 데이터 경로로 바꿀 것
# # # 남자
# # gender = 1
# # male_train = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',\
# #              'down_time', 'up_time', 'similarity', 'grade'])
# # ratio_df = pd.DataFrame(columns = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',\
# #                              'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'])
# # for idx in male_idx:
# #     try:
# #         path = f'train_data_240_json/{idx:03d}/'
# #         result = compare(path, gender)
# #         if label_data[0][idx] == 'A':
# #             result['grade'] = 'A'
# #         elif label_data[0][idx] == 'B':
# #             result['grade'] = 'B'
# #         elif label_data[0][idx] == 'C':
# #             result['grade'] = 'C'
# #         elif label_data[0][idx] == 'F':
# #             result['grade'] = 'F'
# #         male_train.loc[f"{idx:03d}"] = result

# #         out_info, df = cal_ratio(path)
# #         ratio_df = pd.concat([ratio_df, df], axis = 0)
# #     except:
# #         pass

# # male_train_2 = pd.concat([male_train, ratio_df], axis = 1)
# # grade = male_train_2['grade']
# # male_train_2.drop(['grade'], axis = 1, inplace = True)
# # male_train_2['grade'] = grade
# # male_train_2 = male_train_2.dropna()
# # male_train_2.to_csv("male_train_with_frame.csv", index = False)
# # #male_train_2

# # ###########################################나중에 여자 데이터 생성하면 돌릴 부분###########################################
# # # gender = 2
# # # female_train = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',\
# # #              'down_time', 'up_time', 'similarity', 'grade'])
# # # ratio_df = pd.DataFrame(columns = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',\
# # #                              'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'])
# # # for idx in female_idx:
# # #     try:
# # # #         path = f'./new_data/{idx:03d}/'
# # #         result = compare(path, gender)
# # #         if label_data[1][idx] == 'a':
# # #             result['grade'] = 'A'
# # #         elif label_data[1][idx] in ('b', 'a/b'):
# # #             result['grade'] = 'B'
# # #         elif label_data[1][idx] in ('c', 'b/c'):
# # #             result['grade'] = 'C'
# # #         elif label_data[1][idx] in ('f', 'c/f'):
# # #             result['grade'] = 'F'
# # #         female_train.loc[f"{idx:03d}"] = result
        
# # #         out_info, df = cal_ratio(path)
# # #         ratio_df = pd.concat([ratio_df, df], axis = 0)
# # #     except:
# # #         pass

# # # female_train_2 = pd.concat([female_train, ratio_df], axis = 1)
# # # grade = female_train_2['grade']
# # # female_train_2.drop(['grade'], axis = 1, inplace = True)
# # # female_train_2['grade'] = grade
# # # female_train_2 = female_train_2.dropna()
# # # female_train_2.to_csv("female_train_with_frame.csv", index = False)
# # # female_train_2
# # ###########################################나중에 여자 데이터 생성하면 돌릴 부분###########################################


# # Main

# In[24]:


######################################고정부분######################################
# std_path = "new_data/056/"
male_train = pd.read_csv("male_train_with_frame.csv")
male_standard = male_train.loc[male_train.grade == 'A']

# std_path_fe = "new_data/038/"
# female_train = pd.read_csv("female_train_with_frame.csv")
# female_standard = female_train.loc[female_train.grade == 'A']
######################################고정부분######################################

#####################################하기 전에######################################
# 비교할 영상의 경로 -> JSON파일이 담긴 경로
# gender: 1. 남자 // 2. 여자
# visual_comparing 함수의 video 경로 설정
# 기타 함수들의 저장 파일명 설정

#com_path = '/home/centos/ringfit_web/ringfit_ML/000/'
com_path = 'train_data_240_json/104/'


# In[ ]:


ringfit(com_path, 1)


# # 이 위까지 메인 부분 #

# # ML

# ## MLPClassifier - developed

# In[ ]:


# # 예측할 영상 넣기
# # com_path랑 gender만 설정해주고 나머지는 건드릴 필요 X
# com_path = 'train_data_240_json/077/'
# # com_path = 'new_data/010/'
# gender = 1

# test = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',\
#      'down_time', 'up_time', 'similarity'])
# result = compare(com_path, gender)
# output = squat(result, gender)
# test.loc[com_path[-4:-1]] = result

# if gender == 1:
#     train_df = male_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
#     train_x = train_df.iloc[:, :-1]
#     train_y = train_df.iloc[:, -1]

# elif gender == 2:
#     train_df = female_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
#     train_x = female_train.iloc[:, :-1]
#     train_y = female_train.iloc[:, -1]
# else:
#     print("Unavailable Gender")

# out_info, ratio_df = cal_ratio(com_path)
# test = pd.concat([test, ratio_df], axis = 1)
# test['grade'] = None
# # 사용자 영상도 전체 운동 데이터프레임에 추가
# #     if gender == 1:
# #         new_train = pd.concat([male_train, test], axis = 0)
# #     elif gender == 2:
# #         new_train = pd.concat([female_train, test], axis = 0)
# test.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle', 'grade'], axis = 1, inplace = True)

# ## 여기부터 머신러닝
# # import한 모델로 머신러닝 모델 만들어서 돌리면 됨
# predict_result = []
# for _ in range(10):
#     mlp = MLPClassifier(solver = 'lbfgs', max_iter = 500, early_stopping = True)
#     mlp.fit(train_x, train_y)
#     test_x = test 
#     predict_result.append(mlp.predict(test_x)[0])
#     print(mlp.predict(test_x))
#     print(mlp.predict_proba(test_x))
# result_idx, count = 0, 0
# for idx, el in enumerate(['A', 'B', 'C', 'F']):
#     if count < predict_result.count(el):
#         count = predict_result.count(el)
#         result_idx = idx
# final_grade = ['A', 'B', 'C', 'F'][result_idx]
# print(predict_result)
# print(final_grade)


# ## Random Forest

# In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# gender = 1

# test = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',\
#      'down_time', 'up_time', 'similarity'])
# result = compare(com_path, gender)
# output = squat(result, gender)
# test.loc[com_path[-4:-1]] = result

# if gender == 1:
#     train_df = male_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
#     train_x = train_df.iloc[:, :-1]
#     train_y = train_df.iloc[:, -1]

# elif gender == 2:
#     train_df = female_train.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle'], axis = 1)
#     train_x = female_train.iloc[:, :-1]
#     train_y = female_train.iloc[:, -1]
# else:
#     print("Unavailable Gender")

# out_info, ratio_df = cal_ratio(com_path)
# test = pd.concat([test, ratio_df], axis = 1)
# test['grade'] = None
# # 사용자 영상도 전체 운동 데이터프레임에 추가
# #     if gender == 1:
# #         new_train = pd.concat([male_train, test], axis = 0)
# #     elif gender == 2:
# #         new_train = pd.concat([female_train, test], axis = 0)
# test.drop(['RElbow', 'RWrist', 'LElbow', 'LWrist', 'RAnkle', 'LAnkle', 'grade'], axis = 1, inplace = True)

# ##############################################################################################
# gender = 1
# male_test = pd.DataFrame(columns = ['min_angle_lleg', 'min_angle_rleg', 'min_angle_waist',\
#              'down_time', 'up_time', 'similarity', 'grade'])
# ratio_df = pd.DataFrame(columns = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip',\
#                              'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle'])
# for idx in male_test_idx:
#     try:
#         path = f'train_data_240_json/{idx:03d}/'
#         result = compare(path, gender)
#         if label_data[0][idx] == 'A':
#             result['grade'] = 'A'
#         elif label_data[0][idx] == 'B':
#             result['grade'] = 'B'
#         elif label_data[0][idx] == 'C':
#             result['grade'] = 'C'
#         elif label_data[0][idx] == 'F':
#             result['grade'] = 'F'
#         male_test.loc[f"{idx:03d}"] = result

#         out_info, df = cal_ratio(path)
#         ratio_df = pd.concat([ratio_df, df], axis = 0)
#     except:
#         pass
# male_test = pd.concat([male_test, ratio_df], axis = 1)
# grade = male_test['grade']
# male_test.drop(['grade'], axis = 1, inplace = True)
# male_test['grade'] = grade
# male_test = male_test.dropna()

# test_x = male_test.iloc[:, :-1]
# test_y = male_test.iloc[:, -1]

# ##############################################################################################

# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(test_x, test_y)
# y_pred=rf_model.predict(test_x)
# y_pred


# In[ ]:


# pd.crosstab(test_y, y_pred)


# ## XGBOOST

# In[ ]:


# from xgboost import XGBClassifier
# xgb_model=XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=100)
# xgb_model.fit(train_x, train_y)


# In[ ]:


# y_pred=xgb_model.predict(train_x)
# pd.crosstab(train_y,y_pred)


# In[ ]:


# from sklearn.metrics import classification_report
# print(classification_report(train_y, y_pred))


# ## LIGHTGBM

# In[ ]:


# from lightgbm import LGBMClassifier
# lgbm_model=LGBMClassifier(n_estimators=100)
# lgbm_model.fit(train_x.to_numpy(), train_y)


# In[ ]:


# y_pred=lgbm_model.predict(train_x)
# pd.crosstab(train_y, y_pred)


# In[ ]:


# from sklearn.metrics import classification_report
# print(classification_report(train_y, y_pred))


# ## Ensemble

# In[ ]:


# from sklearn.ensemble import VotingClassifier
# voting_model=VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgb', lgbm_model), ('mlp', mlp)], voting='hard')
# voting_model.fit(train_x.to_numpy(), train_y)


# In[ ]:


# voting_model.get_params()


# In[ ]:


# y_pred=voting_model.predict(train_x.to_numpy())
# pd.crosstab(train_y, y_pred)


# # 생성데이터 -> 데이터프레임 생성

# * 훈석 & 정우 -> train
# * 문민 -> test

# In[ ]:




