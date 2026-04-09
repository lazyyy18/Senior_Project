from Library import *


# load data
CsvFile_Path = '9landmarks.csv'
(train_dataset, train4xlabels, train4ylabels, valid_dataset, valid4xlabels, valid4ylabels) = Load_Data4svr(CsvFile_Path)

train_dataset = train_dataset.astype(float)
train4xlabels = train4xlabels.astype(float)
train4ylabels = train4ylabels.astype(float)
valid_dataset = valid_dataset.astype(float)
valid4xlabels = valid4xlabels.astype(float)
valid4ylabels = valid4ylabels.astype(float)

train_dataset = train_dataset[:100000]
train4xlabels = train4xlabels[:100000]
train4ylabels = train4ylabels[:100000]
valid_dataset = valid_dataset[:100000]
valid4xlabels = valid4xlabels[:100000]
valid4ylabels = valid4ylabels[:100000]
print(train4xlabels.shape)





print(type(train4xlabels))

# libsvm-model
#param = svm_parameter('-s 3 -c 4 ')
#-s:esvr -c:cost
m4x = svm_train(train4xlabels, train_dataset,'-s 3 -c 4 ')
p_label, p_acc, p_val = svm_predict(valid4xlabels, valid_dataset, m4x, '-q')
(ACC, MSE, SCC) = evaluations(valid4xlabels, p_label)
print("xMSE : ", MSE)
svm_save_model('save/libsvr4x.h5', m4x)

# libsvm-model
#param = svm_parameter('-s 3 -c 4 ')
#-s:esvr -c:cost
m4y = svm_train(train4ylabels, train_dataset,'-s 3 -c 4 ')
p_label, p_acc, p_val = svm_predict(valid4ylabels, valid_dataset, m4y, '-q')
(ACC, MSE, SCC) = evaluations(valid4ylabels, p_label)
print("yMSE : ", MSE)
svm_save_model('save/libsvr4y.h5', m4y)

'''
m4y = svm_train(train4ylabels, train_dataset,'-s 3' ,'-c 4')
(ACC, MSE, SCC) = evaluations(valid_dataset, valid4ylabels)
svm_save_model('heart_scale.model', m)
'''

'''
# sklearn svr svr4x
svr4x = SVR(kernel='rbf', C=1, epsilon=0.1)
svr4x.fit(train_dataset, train4xlabels)
score = svr4x.evaluate(valid_dataset, valid4xlabels, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

# sklearn svr svr4y
svr4y = SVR(kernel='rbf', C=1, epsilon=0.1)
svr4y.fit(train_dataset, train4ylabels)
score = svr4y.evaluate(valid_dataset, valid4ylabels, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100)) '''

