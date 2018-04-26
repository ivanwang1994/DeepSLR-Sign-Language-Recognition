import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# from tensorflow.contrib import rnn

import numpy as np

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


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 1  # number of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.01
        # 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 1000
        self.batch_size = 300

        # LSTM structure
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 10  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
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

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2] * config.n_layers, state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    # outputs, states = rnn.static_bidirectional_rnn(lstm_cell_1,lstm_cell_2, _X, dtype=tf.float32)


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


if __name__ == "__main__":

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "",
    ]

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


    X_train, y_train = load_X(X_train_signals_paths_emg, X_train_signals_paths_imu,y_train_path_emg,y_train_path_imu)
    X_test=X_train
    y_test=y_train

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train,X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=config1)
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, device_count={"CPU": 11}))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook
