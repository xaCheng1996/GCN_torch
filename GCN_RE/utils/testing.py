import logging
import numpy as np
_logger = logging.getLogger(__name__)

def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sequence = item[0]
        length = len(sequence)
        try:
            size_to_data_dict[length].append(item)
        except:
            size_to_data_dict[length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets

def get_gcn_results(gcn_model, data, maxlength,batch_size, RE_filename, threshold):
    import numpy as np

    import GCN_RE.utils.auxiliary as aux

    true_positive = 0
    pre_sum = 1
    total_sum = 1

    tp_normal = 0
    pre_sum_normal = 1
    total_sum_normal = 1

    tp_overlap = 0
    pre_sum_overlap = 1
    total_sum_overlap = 1

    true_positive_now = 0
    pre_sum_now = 1
    total_sum_now = 1

    total_sentences = 0
    '''
    words, word_vector, word_isEntity, word_index,  relation_entity, relation_vector
    '''
    batch_cnt = 0
    batches = bin_data_into_buckets(data, batch_size=batch_size)
    for batch in batches:
        gcn_batch = []
        relation_set = []
        for item in batch:
            '''
           word_list, word_vector, subj_start, subj_end, obj_start,obj_end, relation_vector
           '''
            # print(len(item))
            words = item[0]
            word_embeddings = item[1]
            subj_start = item[2]
            subj_end = item[3]
            obj_start = item[4]
            obj_end = item[5]
            relation_vector = item[6]
            relation_vector = [np.array(relation_vector)]
            A_fw, A_bw, X = \
                aux.create_graph_from_sentence_and_word_vectors(words, word_embeddings, subj_start, subj_end,
                                                                obj_start, obj_end, maxlength)
            # print(word_is_entity)
            # print(value_matrix_1)
            gcn_batch.append((A_fw, A_bw, X, relation_vector, subj_start, subj_end, obj_start, obj_end))
            # print(gcn_bucket)
            # print(len(gcn_bucket))

        if len(gcn_batch) >= 1:
            prediction = gcn_model.predict(data=gcn_batch, RE_filename = RE_filename)
            # if cnt_batch % 1 == 0:
            threhold = threshold
            relation_set = np.array(relation_set)
            relation_set = np.squeeze(relation_set)
            prediction = np.squeeze(np.array(prediction))
            for lhs, rhs in zip(prediction, relation_set):
                relation_num = str(rhs).count("1")
                lhs = np.array(lhs)
                rhs = np.array(rhs)
                pre = lhs
                for i in range(lhs.shape[0]):
                    pre[i] = sigmoid(lhs[i])
                rhs = rhs.argsort()[-relation_num:][::-1]
                rhs = list(np.sort(rhs))
                logit = []
                # print(str(lhs) + " " + str(rhs))
                for i in range(lhs.shape[0]):
                    pred = sigmoid(lhs[i])
                    # print(pred)
                    if pred > threhold:
                        logit.append(i)

                for i in logit:
                    for j in rhs:
                        if i == j:
                            true_positive += 1
                            true_positive_now += 1
                            if relation_num > 1:
                                tp_overlap += 1
                            else:
                                tp_normal += 1
                            break
                print(str(logit) + " " + str(rhs))
                pre_sum_now += len(logit)
                pre_sum += len(logit)
                total_sum_now += len(rhs)
                total_sum += len(rhs)

                if relation_num > 1:
                    pre_sum_overlap += len(logit)
                    total_sum_overlap += len(rhs)
                else:
                    pre_sum_normal += len(logit)
                    total_sum_normal += len(rhs)

            # print(total_negative)
            if batch_cnt % 100 == 0:
                precision = true_positive_now/(pre_sum_now+1)
                recall = true_positive_now/(total_sum_now+1)
                print("the pre of batch " + str(batch_cnt/100) + " is " + str(true_positive_now/pre_sum_now))
                print("the rec of batch " + str(batch_cnt / 100) + " is " + str(true_positive_now / total_sum_now))
                print("the f1 of batch " + str(batch_cnt / 100) + " is " + str(2 * precision * recall / (precision + recall)))
                true_positive_now = 0
                total_sum_now = 0
                pre_sum_now = 0

    precision_normal = tp_normal / pre_sum_normal
    recall_normal = tp_normal/total_sum_normal
    f1_normal = 2*precision_normal*recall_normal / (precision_normal + recall_normal)
    print("normal, P:R:F = %f %f %f "%(precision_normal, recall_normal, f1_normal))

    # precision_overlap = tp_overlap / pre_sum_overlap
    # recall_overlap = tp_overlap/total_sum_overlap
    # f1_overlap = 2*precision_overlap*recall_overlap / (precision_overlap + recall_overlap)
    # print("overlap, P:R:F = %f %f %f "%(precision_overlap, recall_overlap, f1_overlap))

    precision = true_positive/pre_sum
    recall = true_positive/total_sum
    f1 = 2*precision*recall/(precision + recall)
    return precision, recall, f1


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x_1 = x - np.max(x)
    exp_x = np.exp(x_1)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x