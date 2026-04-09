import csv
import numpy as np
from Library import *
# normalize label 
def normal_labels(labels, weight, height,  new_min=0.2,new_max=0.8):
    normal_labels=[]
    xnormal_labels=float(labels[0]) / float(weight) *(new_max-new_min) + new_min
    ynormal_labels=float(labels[1]) / float(height) *(new_max-new_min) + new_min
    normal_labels.append(xnormal_labels)
    normal_labels.append(ynormal_labels)
    return normal_labels

def normal_data(data, weight, height):
    normal_data=[]
    for i in range(0, 18):
        if i % 2 == 0:
            xnormal_data = float(data[i]) / float(weight) 
            normal_data.append(xnormal_data)
        else: 
            ynormal_data = float(data[i]) / float(height)
            normal_data.append(ynormal_data)
    return normal_data

def normal4xlabels(labels,weight,new_min=0.2,new_max=0.8):
        normal_labels=0
        xnormal_labels=float(labels) / weight *(new_max-new_min) + new_min
        return xnormal_labels

def normal4ylabels(labels,height,new_min=0.2,new_max=0.8):
        normal_labels=0
        ynormal_labels=float(labels) / height *(new_max-new_min) + new_min
        return ynormal_labels
        
  
    
# Load Data
def Load_Data(CsvFile_Path):
    train_dataset=[]
    valid_dataset=[]
    train_labels = []
    valid_labels = []
    i = 1
    t = 1
    #mlp
    file = open('resolution.csv')
    reader = csv.reader(file)
    
    res = []
    for row in reader:
        res.append(row[1:])
    t = 0
    with open(CsvFile_Path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      header = next(rows)
      for row in rows:
        if len(row) > 10:
          tmp1 = []
          tmp2 = []
          if int(row[0]) <= 1200:
             for j in range(1,19):
               tmp1.append(row[j])                   
             train_dataset.append(normal_data(np.array(tmp1, dtype=object), res[i-1][0], res[i-1][1]))
             tmp2.append(row[-2])
             tmp2.append(row[-1])
             train_labels.append(normal_labels(np.array(tmp2, dtype=object), res[i-1][3], res[i-1][2]))
          else:
             for j in range(1,19):
               tmp1.append(row[j])
             valid_dataset.append(normal_data(np.array(tmp1, dtype=object), res[i-1][0], res[i-1][1]))
             tmp2.append(row[-2])
             tmp2.append(row[-1])
             valid_labels.append(normal_labels(np.array(tmp2, dtype=object), res[i-1][3], res[i-1][2]))
             t += 1
        i += 1
        print("count :", i)  
    return np.array(train_dataset, dtype=object), np.array(train_labels, dtype=object), np.array(valid_dataset, dtype=object), np.array(valid_labels, dtype=object)
	


def Load_Data4svr(CsvFile_Path):
    train_dataset=[]
    valid_dataset=[]
    train4xlabels = []
    train4ylabels = []
    valid4xlabels = []
    valid4ylabels = []
    i = 0
    file = open('resolution.csv')
    reader = csv.reader(file1)
    res = []
    for row in reader:
        res.append(row[1:])
    
    #svr
    with open(CsvFile_Path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      header = next(rows)
      for row in rows:
        if len(row) > 10:
          tmp1 = []
          tmp2 = 0
          tmp3 = 0
          if int(row[0]) <= 1200:
            for j in range(1,19):
              tmp1.append(row[j])                   
            train_dataset.append(normal_data(np.array(tmp1, dtype=object), res[i][0], res[i][1]))              
            tmp2 = row[-2]
            train4xlabels.append(normal4xlabels(tmp2,res[i][3]))
            tmp3 = row[-1]
            train4ylabels.append(normal4ylabels(tmp3,res[i][2]))
          else:
              for j in range(1,19):
                tmp1.append(row[j])                   
              train_dataset.append(normal_data(np.array(tmp1, dtype=object), res[i][0], res[i][1]))                 
              tmp2 = row[-2]
              valid4xlabels.append(normal4xlabels(tmp2,res[i][3]))
              tmp3 = row[-1]
              valid4ylabels.append(normal4ylabels(tmp3,res[i][2]))
        i += 1
        print("count: ", i)             
    return np.array(train_dataset, dtype=object), np.array(train4xlabels, dtype=object), np.array(train4ylabels, dtype=object), np.array(valid_dataset, dtype=object), np.array(valid4xlabels, dtype=object), np.array(valid4ylabels, dtype=object)
    
def pixeltomm():
    file = open('screensize.csv')
    reader = csv.reader(file)
    mm = []
    for row in reader:
        mm.append(row)
    return mm    

def convertcameracoord(xth, yth, cvt):
    rvects = cvt[:3]
    tvecs = cvt[3:]
    rvects_= np.array(rvects)
    rvects_ = np.float32(rvects_)
    rvects_, _ = cv2.Rodrigues(rvects_)
    src = np.array([xth, yth, 0]).reshape((1, 3))
    dst = np.dot(src, rvects_ )
    x = dst[0][0] + float(tvecs[0])
    y = dst[0][1] + float(tvecs[1])
    return  x, y

def opencsvfile(path, i):
    file = open(path)
    reader = csv.reader(file)
    tmp = []
    for row in reader:
        tmp.append(row[i:])
    file.close()
    return tmp

def normalize_image_plane(xpts, ypts, camera, distcoeff):
    mat3x3 = np.array(camera).reshape((3,3))
    mat3x3 = np.float32(mat3x3)
    distco = np.float32(distcoeff) 
    src = np.array([float(xpts), float(ypts)])
    dst = cv2.undistortPoints(src, mat3x3, distco)
    x = dst[0][0][0] 
    y = dst[0][0][1]
    return x, y

# Load Data
def Load_testData(CsvFile_Path):
    print("loading")
    #read csv file
    cvt = opencsvfile('monitorpose.csv', 0)
    res = opencsvfile('testrosol.csv', 1)
    camera = opencsvfile('camera.csv', 0)
    distcoeff = opencsvfile('distcoeffs.csv', 0)

    train_dataset=[]
    train_labels = []
    i = 0
    pid = []
    pid.append(0)
    count = 1 # people
    num = 1
    cat = 0
    mm = []
    mm =pixeltomm()
    #mlp
    with open(CsvFile_Path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      header = next(rows)
      for row in rows:
        if len(row) > 60:
          tmp1 = []
          tmp2 = []
          if int(row[0]) != count:
            pid.append(i)
            count += 1
          for j in range(1, 212, 2):
            x, y = normalize_image_plane(row[j], row[j+1], camera[count], distcoeff[count])
            tmp1.append(x)
            tmp1.append(y)
            '''
            if j%2 == 0:
                tmp1.append(row[j]) 
            else :
                tmp1.append(row[j])
            '''
          #train_dataset.append(normal_data(np.array(tmp1, dtype=object),1280,720))
          train_dataset.append(tmp1)
          xth = float(res[num][0]) / float(mm[count][3]) * float(mm[count][1]) 
          yth = float(res[num][1]) / float(mm[count][2]) * float(mm[count][0])
          xth, yth = convertcameracoord(xth, yth, cvt[count])
          tmp2.append(xth/10)
          tmp2.append(yth/10)
          '''
          numbers = re.findall(r'\d+',(row[22]))  # 使用正则表达式提取整数部分
          integers0 = [int(num) for num in numbers]
          numbers = re.findall(r'\d+',(row[21]))  # 使用正则表达式提取整数部分
          integers1 = [int(num) for num in numbers]
          '''
          #train_labels.append(normal_labels(np.array(tmp2, dtype=object),integers0[0], integers1[0]))
          train_labels.append(tmp2)
          i += 1
        num += 1  
    pid.append(i)
    return np.array(pid, dtype=int), np.array(train_dataset, dtype=object), np.array(train_labels, dtype=object)



