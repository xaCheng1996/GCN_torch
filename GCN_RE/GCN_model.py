import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

NAMESPACE = 'GCN'

class TextCNN(nn.Module):
    def __init__(self, output_size, input_dim):
        super(TextCNN, self).__init__()

        self.filter_size = [1,2,3]
        self.filter_num = 64
        self.channel_num = 1
        self.input_dim = input_dim
        self.class_num = output_size

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.channel_num, self.filter_num,
                       (size, self.input_dim)) for size in self.filter_size])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.filter_size) * self.filter_num, self.class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        x_conv = [F.relu(convd(x)).squeeze(3) for convd in self.convs]
        x_pool = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x_conv]
        x_stack = torch.cat(x_pool, 1)
        x_drop = self.dropout(x_stack)
        activation = torch.nn.ReLU()
        x_relu = activation(x_drop)
        logits = self.fc(x_relu)
        # print(logits)
        return logits

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, batch_size):
        result = []
        for i in range(batch_size):
            support = torch.matmul(input[i], self.weight)
            output = torch.matmul(adj[i], support)
            # print(output.shape)
            if self.bias is not None:
                output =  output + self.bias
            activation = torch.nn.LeakyReLU()
            result.append(activation(self.dropout(output)))
        result = np.array(result)
        return result

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers):
        super(BiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(self.hidden_size*2, self.output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.output(out)
        activation = torch.nn.LeakyReLU()
        out = activation(out)
        return out


class GCNReModel(nn.Module):
    _stack_dimension = 2
    _embedding_size = 256
    _output_dim_lstm = 256
    _memory_dim = 256
    _vocab_size = 300
    _hidden_layer1_size = 200
    _hidden_layer2_size = 200
    _output_size = 42
    maxlength = 256

    def __init__(self, batch_size):
        super(GCNReModel, self).__init__()

        # GraphConvolution
        self.batch_size = batch_size
        self.Bi_LSTM = BiRNN(input_size=self._vocab_size, hidden_size=self._memory_dim, output_dim=self._output_dim_lstm, num_layers=2)
        self.DGC = GraphConvolution(self._vocab_size, self._hidden_layer1_size)
        self.EGC = GraphConvolution(self._hidden_layer1_size, self._hidden_layer2_size)
        self.Text_cnn = TextCNN(output_size = self._output_size, input_dim=self._hidden_layer2_size)

    def forward(self, x, aj_matrix_1, aj_matrix_2, subj_start, subj_end, obj_start, obj_end):
        # print(x.shape)
        # LSTM_out = self.Bi_LSTM(x)
        DGC_out = self.DGC(x, aj_matrix_1, self.batch_size)
        EGC_out = self.EGC(DGC_out, aj_matrix_2, self.batch_size)
        prediction = []
        for i in range(EGC_out.shape[0]):
            subj = EGC_out[i][subj_start:subj_end+1, :]
            obj = EGC_out[i][obj_start:obj_end+1, :]
            entity_pair = torch.cat((subj, obj), 0)
            entity_pair_full = torch.cat((entity_pair, DGC_out[i][:, :]), 0)
            prediction.append(entity_pair_full)
        prediction_t = torch.stack(prediction)
        # print(prediction_t.shape)
        logits = self.Text_cnn(prediction_t)
        # print(logits.shape)
        # prediction_tensor = torch.cat(logits)
        return logits


class Train_and_E(object):
    def __init__(self, maxlength):
        self.maxlength = maxlength
        self.batch_size = 32
    def __train(self, A_GCN_l1,A_GCN_l2, X, y, subj_start, subj_end, obj_start, obj_end):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])

        # print(len(X))
        X_array = np.array(X)
        y_array = np.squeeze(np.array(y))
        # y_array = np.transpose(y_array, (1, 0, 2))
        self.batch_size = X_array.shape[0]
        X_array = np.squeeze(X_array)


        Aj_matrix_1 = torch.from_numpy(Aj_matrix_1).float().cuda()
        Aj_matrix_2 = torch.from_numpy(Aj_matrix_2).float().cuda()
        X2 = torch.from_numpy(X_array).float().cuda()
        y_array = torch.from_numpy(y_array).float().cuda()

        Aj_matrix_1_gen = []
        for item in Aj_matrix_1:
            Aj_matrix_1_gen.append(self.gen_adj(item))

        Aj_matrix_2_gen = []
        for item in Aj_matrix_2:
            Aj_matrix_2_gen.append(self.gen_adj(item))

        if len(X) == 1:
            X2 = torch.unsqueeze(X2, 0)
            y_array = torch.unsqueeze(y_array, 0)

        subj_start = int(subj_start[0])
        subj_end = int(subj_end[0])
        obj_start = int(obj_start[0])
        obj_end = int(obj_end[0])

        self.model = GCNReModel(self.batch_size).cuda()

        # pos_weight = torch.ones([42])
        criterion = nn.CrossEntropyLoss().cuda()

        for item in self.model.parameters():
            print(item)

        optimizer = torch.optim.AdamW(self.model.parameters())

        prediction = self.model(X2, Aj_matrix_1_gen, Aj_matrix_2_gen, subj_start, subj_end, obj_start, obj_end)

        # prediction = torch.sigmoid(prediction)
        loss = criterion(prediction, torch.argmax(y_array, 1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        target = torch.argmax(prediction, 1)
        correct = 0
        # print(target)
        correct += (target == torch.argmax(y_array, 1)).sum().float()

        return correct, loss

    def train(self, data):
        correct, loss = self.__train([data[i][0] for i in range(len(data))],
                            [data[i][1] for i in range(len(data))],
                            [data[i][2] for i in range(len(data))],
                            [data[i][3] for i in range(len(data))],
                            [data[i][4] for i in range(len(data))],
                            [data[i][5] for i in range(len(data))],
                            [data[i][6] for i in range(len(data))],
                            [data[i][7] for i in range(len(data))],
                           )


        # print("acc: " + str(acc))
        return correct, loss

    def _predict(self, A_GCN_l1,A_GCN_l2, X, y, subj_start, subj_end, obj_start, obj_end, RE_filename):
        Aj_matrix_1 = np.array([item for item in A_GCN_l1])
        Aj_matrix_2 = np.array([item for item in A_GCN_l2])

        X_array = np.array(X)
        y_array = np.squeeze(np.array(y))
        # y_array = np.transpose(y_array, (1, 0, 2))
        self.batch_size = X_array.shape[0]

        Aj_matrix_1 = torch.from_numpy(Aj_matrix_1).float().cuda()
        Aj_matrix_2 = torch.from_numpy(Aj_matrix_2).float().cuda()
        X2 = torch.from_numpy(X_array).float().cuda()
        y_array = torch.from_numpy(y_array).float().cuda()

        subj_start = int(subj_start[0])
        subj_end = int(subj_end[0])
        obj_start = int(obj_start[0])
        obj_end = int(obj_end[0])

        self.model = GCNReModel(self.batch_size).cuda()
        self.model.load_state_dict(torch.load(RE_filename))
        prediction = self.model(X2, Aj_matrix_1, Aj_matrix_2, subj_start, subj_end, obj_start, obj_end)
        return prediction

    def predict(self, data, RE_filename):
        # outputs = np.array(self._predict([A_fw], [A_bw], [X], [value_matrix], [A_fw_dig], [A_bw_dig]))
        outputs = self._predict([data[i][0] for i in range(len(data))],
                            [data[i][1] for i in range(len(data))],
                            [data[i][2] for i in range(len(data))],
                            [data[i][3] for i in range(len(data))],
                            [data[i][4] for i in range(len(data))],
                            [data[i][5] for i in range(len(data))],
                            [data[i][6] for i in range(len(data))],
                            [data[i][7] for i in range(len(data))],
                                RE_filename,
                           )
        prediction = []
        for item in outputs:
            prediction.append(item)
        return prediction

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def gen_adj(self, A):
        # print(A)
        # print(A.sum(1))
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(D, A), D)
        return adj