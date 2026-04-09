from mat4py import loadmat
from Library import *

path = "../MPIIFaceGaze/MPIIFaceGaze"
# creat .csv file
csvfile = open('monitorpose.csv', 'w', newline='')
writer = csv.writer(csvfile, delimiter = ',')

# write title
title = []
title.append('rvecs')
title.append('tvecs')
writer.writerow(title)
i = 0

for cat in sorted(os.listdir(path)):
    if cat != 'readme.txt':
        cat_ = os.path.join(path, cat)
        for cat__ in sorted(os.listdir(cat_)):
            if cat__ == 'Calibration':
                calib_path = os.path.join(cat_, cat__)
                for calib_ in os.listdir(calib_path):
                    hw = [] # height and width
                    if calib_ == 'monitorPose.mat':
                        data = loadmat(calib_path+'/monitorPose.mat')
                        for i in range(0, 3):
                            hw.append(data['rvects'][i][0])
                        for j in range(0, 3):
                            hw.append(data['tvecs'][j][0])
                        writer.writerow(hw)

'''

for cat in sorted(os.listdir(path)):
    if cat != 'readme.txt':
        cat_ = os.path.join(path, cat)
        for cat__ in sorted(os.listdir(cat_)):
            if cat__ == 'Calibration':
                calib_path = os.path.join(cat_, cat__)
                for calib_ in os.listdir(calib_path):
                    hw = [] # height and width
                    if calib_ == 'monitorPose.mat':
                        data = loadmat(calib_path+'/monitorPose.mat')
                        for i in range(0, 3):
                            hw.append(data['tvecs'][i][0])
                        writer.writerow(hw)
'''
