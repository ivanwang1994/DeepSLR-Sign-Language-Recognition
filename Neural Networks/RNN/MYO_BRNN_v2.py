from __future__ import print_function
from tensorflow.contrib import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import random


config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True

# Load "X" (the neural network's training and testing inputs)

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

    # 4. 512 * x check
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

    # 6. change 1*x*512*24 to 24*x*512
    _X_signals=[]
    for i in range(24):
        _serie_every_24=[]
        for serie in X_signals:
            _serie_every_move=[]
            for j in range(256):
                _serie_every_move.extend([serie[i+j*24]])
            _serie_every_24.append(_serie_every_move)
        _X_signals.append(_serie_every_24)

    return np.transpose(np.array(np.array(_X_signals,dtype=np.float32)), (1, 2, 0)), np.array(y,dtype=np.float32)


def data_segmentation(X,y):
    X_range=range(0,len(X))
    resultList = random.sample(X_range,int(len(X)*2/10))
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    for i in X_range:
        if (i in resultList):
            X[i]
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return np.array(X_train,dtype=np.float32),np.array(y_train,dtype=np.float32),np.array(X_test,dtype=np.float32),np.array(y_test,dtype=np.float32)

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 2  # number of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        # 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 1000
        self.batch_size = 150
        self.output_keep_prob = 0.5

        # LSTM structure
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 10  # Final output classes


        # define weights name
        with tf.name_scope('weights'):
            self.W = {
                    'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden]), name = 'W'),
                    'output': tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]),name = 'W')
            }
            tf.summary.histogram('output_layer_weights', self.W['hidden'])
            tf.summary.histogram('output_layer_weights_1', self.W['output'])
        # define biase
        with tf.name_scope('Wx_plus_b'):
                self.biases = {
                        'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
                        'output': tf.Variable(tf.random_normal([self.n_classes]))
                }
                tf.summary.histogram('output_layer_biases', self.biases['hidden'])
                tf.summary.histogram('output_layer_weights_1', self.W['output'])

      


def LSTM_Network(_X, config):
    
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

#    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
#    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
#    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    
    
    
    def lstm_cell(config):
        cell = rnn.LSTMCell(config.n_hidden)
        with tf.name_scope('lstm_dropout'):
            return rnn.DropoutWrapper(cell, output_keep_prob=config.output_keep_prob)
        # _X = tf.unstack(_X, config.n_steps, 1)

        # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        # 实现多层 LSTM
        # [attn_cell() for _ in range(n_layers)]
        # enc_cells = []
        # for i in range(0, self.n_layers):
        #     enc_cells.append(lstm_cell())
        # with tf.name_scope('lstm_cells_layers'):
        #     mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)



    with tf.name_scope("cell_1"):
        lstm_cell_1 = tf.contrib.rnn.MultiRNNCell([lstm_cell(config) for _ in range(config.n_layers)], state_is_tuple = True)

    # Backward direction cell
    with tf.name_scope("cell_2"):
        lstm_cell_2 = tf.contrib.rnn.MultiRNNCell([lstm_cell(config) for _ in range(config.n_layers)], state_is_tuple = True)

        # # 全零初始化 state
        # _init_state = lstm_cell.zero_state(config.batch_size, dtype=tf.float32)
        # # dynamic_rnn 运行网络
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, _X, initial_state=_init_state, dtype=tf.float32,
        #                                     time_major=False)
        # # 输出
        # # return tf.matmul(outputs[:,-1,:], Weights) + biases
        # return tf.nn.softmax(tf.matmul(outputs[:, -1, :], config.W) + config.biases)




    
    
    
#    lstm_cell_1 = tf.contrib.rnn.MultiRNNCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
#    # Backward direction cell
#    lstm_cell_2 = tf.contrib.rnn.MultiRNNCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    
#    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2] * config.n_layers, state_is_tuple=True)
    # Get LSTM cell output
#    outputs, states = rnn.static_bidirectional_rnn(lstm_cell_1,lstm_cell_2, _X, dtype=tf.float32)
    
    
    
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_cell_1, lstm_cell_2, _X,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_cell_1, lstm_cell_2, _X,
                                        dtype=tf.float32)
    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
#    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(outputs[-1], config.W['output']) + config.biases['output']


def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def main():
    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised input features for the neural network
#    INPUT_SIGNAL_TYPES = [
#        "",
#    ]

#    DATA_PATH = "data/"
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

    X_all, y_all = load_X(X_train_signals_paths_emg, X_train_signals_paths_imu,y_train_path_emg,y_train_path_imu)
    X_train, y_train, X_test, y_test = data_segmentation(X_all,y_all)
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    # ---------------
    # --------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and "
          "normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name='x_input')
        Y = tf.placeholder(tf.float32, [None, config.n_classes], name='y_input')
        tf.output_keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')  # 保持多少不被 dropout
        tf.batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')  # 批大小

    with tf.name_scope('output_layer'):
        pred_Y = LSTM_Network(X, config)
        tf.summary.histogram('outputs', pred_Y)
    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
        tf.summary.scalar('loss', cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate).minimize(cost)
        # tf.summary.scalar('optimizer', optimizer)

    # with tf.name_scope('optimizer'):
    #     correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    # with tf.name_scope('accuracy'):
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    #     tf.summary.scalar('accuracy', accuracy)



    # correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    # accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, axis=1), predictions=tf.argmax(pred_Y, axis=1))[1]
        tf.summary.scalar('accuracy', accuracy)
    
    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=config1)
    merged = tf.summary.merge_all()
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, device_count={"CPU": 11}))
    # init = tf.global_variables_initializer()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init)




    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter("logs-myo/", sess.graph)

        best_accuracy = 0.0
        saver = tf.train.Saver(max_to_keep=1)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # Start training for each batch and loop epochs
        for i in range(config.training_epochs):
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: y_train[start:end]})
                # if i % 50 == 0:
                #     result = sess.run(merged,
                #                       feed_dict={X: X_train[start:end],
                #                                  Y: y_train[start:end]})
                #     writer.add_summary(result, i)




            # Test completely at every epoch: calculate accuracy
            result, pred_out, accuracy_out, loss_out = sess.run(
                [merged, pred_Y, accuracy, cost],
                feed_dict={
                    X: X_test,
                    Y: y_test
                },
                options=run_options,
                run_metadata=run_metadata
            )
            writer.add_summary(result, i)

            print("traing iter: {},".format(i) +
                  " test accuracy : {},".format(accuracy_out) +
                  " loss : {}".format(loss_out))

            best_accuracy = max(best_accuracy, accuracy_out)
            if best_accuracy < accuracy_out:
                best_accuracy = accuracy_out
                saver.save(sess, "Model/model.ckpt")
        writer.close()


        print("")
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("")

if __name__ == "__main__":
    main()