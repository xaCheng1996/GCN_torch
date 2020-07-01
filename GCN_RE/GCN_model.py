import torch
import numpy as np
import torch.nn as nn


NAMESPACE = 'GCN'
_random_seed = 3141592653589

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim,support,act_func=None,featureless=False,dropout_rate=0.,bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))

            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        return out


class GCNReModel(nn.Module):
    _stack_dimension = 2
    _embedding_size = 256
    _memory_dim = 256
    _vocab_size = 300
    _hidden_layer1_size = 200
    _hidden_layer2_size = 200
    _output_size = 42
    loss = 0

    def __init__(self, aj_matrix_1, aj_matrix_2, dropout_rate=0.):
        super(GCNReModel, self).__init__()

        # GraphConvolution
        self.Bi_LSTM = BiRNN(self._vocab_size, hidden_size=self._memory_dim, num_layers=2)
        self.DGC = GraphConvolution(self._memory_dim, self._hidden_layer1_size, aj_matrix_1, act_func=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.EGC = GraphConvolution(self._hidden_layer1_size, self._hidden_layer2_size, aj_matrix_2, dropout_rate=dropout_rate)
        self.fc = nn.Linear(self._hidden_layer2_size, self._output_size)

    def forward(self, x):
        LSTM_out = self.Bi_LSTM(x)
        DGC_out = self.DGC(LSTM_out)
        EGC_out = self.EGC(DGC_out)
        prediction = self.fc(EGC_out)
        return prediction


class Train_and_E(object):
    def __init__(self, maxlength):
        self.maxlength = maxlength
    def __train(self, A_GCN_l1,A_GCN_l2, X, y, value_matrix,A_GCN_dig_1, A_GCN_dig_2):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])
        aj_matrix_dig_1 = np.array([item for item in A_GCN_dig_1])
        aj_matrix_dig_2 = np.array([item for item in A_GCN_dig_2])

        X_array = np.array(X)
        X2 = np.copy(X_array)
        y_array = np.array(y)

        A_GCN_dig_1_x = np.array([self.matrix_pow(item) for item in aj_matrix_dig_1])
        A_GCN_dig_2_x = np.array([self.matrix_pow(item) for item in aj_matrix_dig_2])

        # print("the shape of u in _train:")
        # print(np.array(y_array).shape)

        X_array = np.transpose(X_array, (1, 0, 2))
        X2 = np.transpose(X2, (1, 0, 2))
        X2 = X2[::-1, :, :].copy()

        y_array = np.transpose(y_array, (1, 0, 2))

        A_GCN_l1 = torch.from_numpy(Aj_matrix_1)
        A_GCN_l2 = torch.from_numpy(Aj_matrix_2)
        X2 = torch.from_numpy(X2)
        y_array = torch.from_numpy(y_array)
        A_GCN_dig_1_x = torch.from_numpy(A_GCN_dig_1_x)
        A_GCN_dig_2_x = torch.from_numpy(A_GCN_dig_2_x)

        model = GCNReModel(A_GCN_l1, A_GCN_l2)
        # print(value_matrix)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        prediction = model(X2)
        loss = criterion(prediction, torch.from_numpy(y_array))
        acc = 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, acc

    def train(self, data, maxlength):
        loss, acc = self.__train([data[i][0] for i in range(len(data))],
                            [data[i][1] for i in range(len(data))],
                            [data[i][2] for i in range(len(data))],
                            [data[i][3] for i in range(len(data))],
                            [data[i][4] for i in range(len(data))],
                            [data[i][5] for i in range(len(data))],
                            [data[i][6] for i in range(len(data))],
                           )


        print("acc: " + str(acc))
        return loss

    def _predict(self, A_GCN_l1,A_GCN_l2, X, value_matrix, A_fw_dig, A_bw_dig):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])
        aj_matrix_dig_1 = np.array([item for item in A_fw_dig])
        aj_matrix_dig_2 = np.array([item for item in A_bw_dig])
        value_matrix_t = np.array([item for item in value_matrix])

        X_in = np.array(X)
        X2 = np.copy(X)

        A_GCN_dig_1_x = np.array([self.matrix_pow(item) for item in aj_matrix_dig_1])
        A_GCN_dig_2_x = np.array([self.matrix_pow(item) for item in aj_matrix_dig_2])

        # print("the shape of X in _train:")
        # print(np.array(X2).shape)

        # print(X_in.shape)
        X_in = np.transpose(X_in, (1, 0, 2))
        X2 = np.transpose(X2, (1, 0, 2))
        X2 = X2[::-1, :, :]


        feed_dict = {}
        feed_dict.update({self.word_input: X_in})
        feed_dict.update({self.word_input_bw: X2})

        feed_dict.update({self.aj_matrix_1: Aj_matrix_1})
        feed_dict.update({self.aj_matrix_2: Aj_matrix_2})
        feed_dict.update({self.value_matrix: value_matrix_t})
        feed_dict.update({self.aj_matrix_1_dig: A_GCN_dig_1_x})
        feed_dict.update({self.aj_matrix_2_dig: A_GCN_dig_2_x})


        y_batch = self.sess.run([self.prediction], feed_dict)
        return y_batch

    def predict(self, data):
        # outputs = np.array(self._predict([A_fw], [A_bw], [X], [value_matrix], [A_fw_dig], [A_bw_dig]))
        outputs = np.array(self._predict(
                                [data[i][0] for i in range(len(data))],
                                [data[i][1] for i in range(len(data))],
                                [data[i][2] for i in range(len(data))],
                                [data[i][3] for i in range(len(data))],
                                [data[i][4] for i in range(len(data))],
                                [data[i][5] for i in range(len(data))],
                               ))
        prediction = []
        for item in outputs:
            prediction.append(item)
        return prediction

    def save(self, filename):
        saver = tf.train.Saver()
        # print(self.sess)
        saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        # saver = tf.train.Saver([v for v in tf.global_variables() if NAMESPACE in v.name])
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)

    def matrix_pow(self, matrix):
        matrix_new = np.array(matrix)
        for i in range(matrix_new.shape[0]):
            if matrix_new[i][i] == 0: continue
            matrix_new[i][i] = (matrix_new[i][i])**(-0.5)
        return matrix_new