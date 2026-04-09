from Library import *

def pixeltomm():
    file = open('screensize.csv')
    reader = csv.reader(file)
    mm = []
    for row in reader:
        mm.append(row)
    return mm


CsvFile_Path = '68test.csv'
#(train_dataset, train_labels, valid_dataset, valid_labels) = Load_Data(CsvFile_Path)
(pid, train_dataset, train_labels) = Load_testData(CsvFile_Path)
train_dataset = train_dataset.astype(float)
train_labels = train_labels.astype(float)
print(train_dataset)


print(train_dataset.shape)
print(train_labels.shape)


seed = 7
np.random.seed(seed)

#tensorflow
model = Sequential([
        Dense(64, kernel_initializer=HeNormal(), input_shape=(212,), activation='LeakyReLU'),
        Dropout(0),
        BatchNormalization(),
        Dense(128, kernel_initializer=HeNormal(), activation='LeakyReLU'),
        Dropout(0),
        BatchNormalization(),
        Dense(256, kernel_initializer=HeNormal(), activation='LeakyReLU'),
        Dropout(0),
        BatchNormalization(),
        Dense(512, kernel_initializer=HeNormal(), activation='LeakyReLU'),
        Dropout(0),
        BatchNormalization(),
        Dense(2, kernel_initializer= HeNormal(), activation='linear'),
        ])

# init optimizers 
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
# lr and wd can be a function or a tensor
lr = 9e-2
wd = lambda: 3e-4 * schedule(step)

opt = tfa.optimizers.SGDW(
    learning_rate=lr, weight_decay=wd, momentum=0.9)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

#將模型畫成圖片, 並存為png檔
tf.keras.utils.plot_model(model, to_file='MLP.png', show_shapes=True)
model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.7, weight_decay=0.5), loss=tf.keras.losses.MeanSquaredError() ,metrics=[RootMeanSquaredError()])
#model.compile(optimizer=opt, loss=Huber(delta=13) , metrics=[RootMeanSquaredError()])
#model.compile(optimizer=opt, loss=root_mean_squared_error, metrics=[RootMeanSquaredError()])

#trans pixel to mm
mm = 0
# init
traindata = []
trainlabel = []
testdata = []                   
testlabel = []
model_name = 'MLP'
res = 0
print(mm)
# training
for i in range(0, 15):
    #leave one person out
    traindata = np.concatenate((train_dataset[:pid[i]], train_dataset[pid[i+1]:]))
    trainlabel = np.concatenate((train_labels[:pid[i]], train_labels[pid[i+1]:]))
    testdata = train_dataset[pid[i]:pid[i+1]]
    testlabel = train_labels[pid[i]:pid[i+1]]
    print(i+1, " person leave")
    #train
    history = model.fit(traindata, trainlabel, shuffle=True, validation_data=(testdata, testlabel), epochs=20, batch_size=32)
    score = model.evaluate(testdata, testlabel, verbose = 0)
    print("%s: %f" % (model.metrics_names[1], score[1]))
    preds = model.predict(testdata)
    res = 0
    for j in range(0, pid[i+1]-pid[i]):
        x = (preds[j][0] - testlabel[j][0]) 
        y = (preds[j][1] - testlabel[j][1]) 
        res  += pow(pow(x, 2) + pow(y, 2),0.5)
    print( res / (pid[i+1]-pid[i]))
    mm = mm + (res / (pid[i+1]-pid[i]))
    if i != 15:
        for j in range(0, 20):
            tmp = j
            print("true :", j, "= ", testlabel[tmp])
            print("preds:", j, "= ", preds[tmp])
    plot_loss(history, score, i+1, model_name) 

# evaluate mse && plot distribution
preds = model.predict(train_dataset)
plot_distribution(preds, train_labels)
mm = mm/15
print(mm)
model.summary()
#model_file = 'save/batch128_ep100_RMSE{}%.h5'.format(score[1] * 100)
#model.save_weights(model_file)
#plot_pred(preds, valid_labels, score)


