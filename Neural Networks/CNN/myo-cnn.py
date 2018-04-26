import tensorflow as tf
import numpy as np
import os
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def load_X(X_signals_paths_emg, X_signals_paths_imu, y_path_emg, y_path_imu):
    X_signals_emg = []
    X_signals_imu = []
    X_signals = []
    move_data = []

    # 1. change to list in X_signals_XXX
    # emg-----------------------
    for signal_type_path in X_signals_paths_emg:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        move_data.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split(' ') for row in file
            ]]
        )
        for serie in move_data[0]:
            serie = serie.tolist()
            X_signals_emg.append(serie)
        #
        file.close()

    # imu-----------------------
    move_data = []
    for signal_type_path in X_signals_paths_imu:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        move_data.append(
            np.array(serie, dtype=np.float32) for serie in [
                row.strip().split(' ') for row in file
            ]
        )
        for serie in move_data[0]:
            serie = serie.tolist()
            X_signals_imu.append(serie)
        #
        file.close()

    # 2. pedding emg block according to imu y
    # make the emg table
    y_emg_table=[]
    y_emg=[]
    file = open(y_path_emg, 'r')
    y_emg.extend(row.strip().split(' ')[0] for row in file)
    file.close()
    _serie=y_emg[0]
    repeat_num=1
    num=1
    for serie in y_emg:
        if num!=1:
            if serie==_serie:
                repeat_num=repeat_num+1
                if y_emg_table.__len__()>1:
                    if (y_emg_table[y_emg_table.__len__()-1][2]+repeat_num == y_emg.__len__()-1):
                        y_emg_table.append([_serie, y_emg_table[y_emg_table.__len__() - 1][2] + 1,y_emg_table[y_emg_table.__len__() - 1][2] + repeat_num])
            else:
                if y_emg_table.__len__()==0:
                    y_emg_table.append([_serie, 0, repeat_num-1])
                else:
                    y_emg_table.append([_serie,y_emg_table[y_emg_table.__len__()-1][2]+1,y_emg_table[y_emg_table.__len__()-1][2]+repeat_num])
                _serie=serie
                repeat_num=1
        else:
            num=num+1
    # make the imu table
    y_imu_table = []
    y_imu = []
    file = open(y_path_imu, 'r')
    y_imu.extend(row.strip().split(' ')[0] for row in file)
    file.close()
    _serie = y_imu[0]
    repeat_num = 1
    num = 1
    for serie in y_imu:
        if num != 1:
            if serie == _serie:
                repeat_num = repeat_num + 1
                if y_imu_table.__len__() > 1:
                    if (y_imu_table[y_imu_table.__len__() - 1][2] + repeat_num == y_imu.__len__() - 1):
                        y_imu_table.append([_serie, y_imu_table[y_imu_table.__len__() - 1][2] + 1,
                                            y_imu_table[y_imu_table.__len__() - 1][2] + repeat_num])
            else:
                if y_imu_table.__len__() == 0:
                    y_imu_table.append([_serie, 0, repeat_num - 1])
                else:
                    y_imu_table.append([_serie, y_imu_table[y_imu_table.__len__() - 1][2] + 1,
                                        y_imu_table[y_imu_table.__len__() - 1][2] + repeat_num])
                _serie = serie
                repeat_num = 1
        else:
            num = num + 1
    #extend according to table
    if y_emg_table.__len__()!=y_imu_table.__len__():
        print("emg.table_num!=imu.table_num!")
        exit(0)
    for num in range(y_emg_table.__len__()):
        if y_imu_table[num][0]!=y_emg_table[num][0]:
            print("emg.table is not according to imu.table!")
            exit(0)
        diff_num=y_imu_table[num][2]-y_emg_table[num][2]
        if diff_num>0:
            extend_move=X_signals_emg[y_emg_table[num][2]]
            place_num=y_emg_table[num][2]+1
            for i in range(diff_num):
                X_signals_emg.insert(place_num,extend_move)
                place_num=place_num+1

            y_emg_table[num][2]=y_emg_table[num][2]+diff_num
            for i in range(num+1,y_emg_table.__len__()):
                y_emg_table[i][2] = y_emg_table[i][2] + diff_num
                y_emg_table[i][1] = y_emg_table[i][1] + diff_num
        elif diff_num<0:
            diff_num=-diff_num
            extend_move = X_signals_imu[y_imu_table[num][2]]
            place_num = y_imu_table[num][2] + 1
            for i in range(diff_num):
                X_signals_imu.insert(place_num, extend_move)
                place_num = place_num + 1

            y_imu_table[num][2] = y_imu_table[num][2] + diff_num
            for i in range(num + 1, y_imu_table.__len__()):
                y_imu_table[i][2] = y_imu_table[i][2] + diff_num
                y_imu_table[i][1] = y_imu_table[i][1] + diff_num

    print(
        "Number of emg and imu block is " + X_signals_emg.__len__().__str__() + " " + X_signals_imu.__len__().__str__())

    # 3. rewirte the responding y
    y=[]
    for i in range(y_emg_table.__len__()):
        y.extend([y_emg_table[i][0]] for k in range(y_emg_table[i][2]-y_emg_table[i][1]+1))

    # 4. 256 * x check
    for i in range(X_signals_emg.__len__()):
        if len(X_signals_emg[i])>256*16:
            for j in range(y_emg_table.__len__()):
                if y[j]==y[i] and len(X_signals_emg[j])<256*16:
                    X_signals_emg[i]=X_signals_emg[j]
                    break
        if len(X_signals_emg[i]) < 256 * 16:
            X_signals_emg[i].extend([0 for k in range(256*16-len(X_signals_emg[i]))])

    for i in range(X_signals_imu.__len__()):
        if len(X_signals_imu[i]) > 256 * 8:
            for j in range(y_imu_table.__len__()):
                if y[j] == y[i] and len(X_signals_imu[j]) < 256 * 8:
                    X_signals_imu[i] = X_signals_imu[j]
                    break
        if len(X_signals_imu[i]) < 256 * 8:
            X_signals_imu[i].extend([0 for k in range(256 * 8 - len(X_signals_imu[i]))])

    # 5. connection:16-8
    for i in range(y.__len__()):
        serie = []
        for j in range(256):
            serie.extend([X_signals_emg[i][k] for k in range(j * 16,(j+1) * 16)])
            serie.extend([X_signals_imu[i][k] for k in range(j * 8, (j + 1) * 8)])
        X_signals.append(serie)

# 6. change 1*x*256*24 to 24*x*256
    _X_signals=[]
    for i in range(24):
        _serie_every_24=[]
        for serie in X_signals:
            _serie_every_move=[]
            for j in range(256):
                _serie_every_move.extend([serie[i+j*24]])
            _serie_every_24.append(_serie_every_move)
        _X_signals.append(_serie_every_24)

    _X_signals = [_X_signals]
    return np.transpose(np.array(np.array(_X_signals,dtype=np.float32)), (2,1,3,0)), np.array(y,dtype=np.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def gettest(X_train,train_label):
    X_test=[]
    test_label=[]
    for i in range(300):
        k = random.randint(1, 1815)
        X_test.append(X_train[k])
        test_label.append(train_label[k])
    return np.array(X_test,dtype=np.float32),np.array(test_label,dtype=np.float32)

batch_size = 512
X_ = tf.placeholder(tf.float32, [None, 24, 256, 1],name='cnn_X')
label_ = tf.placeholder(tf.float32, [None, 10],name='cnn_Y')

#input shape [batch, in_height, in_width, in_channels]
#kernel shape [filter_height, filter_width, in_channels, out_channels]
'''
	1*9
	stride = 2 
	padding
	24*256->24*128*32
'''
W_conv1 = weight_variable([1, 9, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(X_, W_conv1, strides=[1, 1, 2, 1], padding='SAME') + b_conv1)
'''
	pooling
	24*128*32->24*64*32
'''
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
'''
	1*3
	stride = 1
	24*64*32->24*64*64
'''
W_conv2 = weight_variable([1, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
'''
	1*3
	stride = 1
	padding
	24*64*64->24*64*128
'''

W_conv3 = weight_variable([1, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(
    tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

'''
'''
'''
	pooling
	24*64*128->24*32*128
'''
h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='VALID')
'''
	6*1
	24*32*128->1*32*128
'''
W_conv4 = weight_variable([24, 1, 128, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(
    tf.nn.conv2d(h_pool2, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)

'''
	input flat 32*128=4096
	output 118
'''


h_flat = tf.contrib.layers.flatten(h_conv4)
cnn_output = tf.multiply(h_conv4,1,name='cnn_output')
W_fc1 = weight_variable([4096, 256])
b_fc1 = bias_variable([256])
h_fc1 = tf.nn.softmax(tf.matmul(h_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(y_conv+1e-10), reduction_indices=[1]),name='cnn_loss')
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label_, 1),name='cnn_pre_Y')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='cnn_accuracy')

DATA_PATH = "data/"
DATASET_PATH = "data/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)
TRAIN = "train/"
TEST = "test/"
X_train_signals_paths_emg = [
    DATASET_PATH + TRAIN + "x_data/emg/x_emg.txt"
]
X_train_signals_paths_imu = [
    DATASET_PATH + TRAIN + "x_data/imu/x_imu.txt"
]
X_test_signals_paths_emg = [
    DATASET_PATH + TEST + "x_data/emg/x_emg.txt"
]
X_test_signals_paths_imu = [
    DATASET_PATH + TEST + "x_data/imu/x_imu.txt"
]
y_train_path_emg = DATASET_PATH + TRAIN + "y_emg.txt"
y_train_path_imu = DATASET_PATH + TRAIN + "y_imu.txt"
y_test_path_emg = DATASET_PATH + TEST + "y_emg.txt"
y_test_path_imu = DATASET_PATH + TEST + "y_imu.txt"
# -----------------------------------------------------------------

X_train, train_label = load_X(X_train_signals_paths_emg, X_train_signals_paths_imu, y_train_path_emg, y_train_path_imu)
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))
print(len(X_train[0][0][0]))
X_test,test_label=gettest(X_train, train_label)


train_label = one_hot(train_label)
test_label = one_hot(test_label)

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

best_accuracy = 0
for i in range(1000):
    l = len(train_label)
    batch_idxs = int(l / batch_size)
    index = list(range(l))
    random.shuffle(index)
    for idx in range(batch_idxs):
        image_idx = X_train[index[idx * batch_size:(idx + 1) * batch_size]]
        label_idx = train_label[index[idx * batch_size:(idx + 1) * batch_size]]
        #print(start,end)
        acc, loss, _ = sess.run([accuracy, cross_entropy, train_step], feed_dict={
            X_: image_idx,
            label_: label_idx
        })
        if idx % 100 == 0:
            print(str(i) + 'th cross_entropy:', str(loss), 'train_accuracy:', str(acc))
        # Test completely at every epoch: calculate accuracy
    accuracy_out, loss_out = sess.run(
        [accuracy, cross_entropy],
        feed_dict={
            X_: X_test,
            label_: test_label
        }
    )
    if accuracy_out > best_accuracy:
        best_accuracy = accuracy_out
    print(str(i)+'th cross_entropy:', str(loss_out), 'test_accuracy:', str(accuracy_out))

print("best accuracy:"+str(best_accuracy))
