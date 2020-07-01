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


def train_and_save(dataset, saving_dir, data_name, epochs, bucket_size):
    import random
    import sys
    import pickle
    import numpy as np
    import GCN_RE.utils.auxiliary as aux

    import GCN_RE
    from .auxiliary import  create_full_sentence

    print("dataset is ready, now we are taking them from file")
    sentences= aux.get_all_sentence(dataset, data_name)
    maxlength = 256
    print('Computing the transition matrix')
    data = aux.get_data_from_sentences(sentences, maxlength)
    print("we resize the dataset, now we will build dependency tree and train them")
    buckets = bin_data_into_buckets(data, bucket_size)
    gcn_train = GCN_RE.GCN_model.Train_and_E(maxlength)

    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        cnt_batch = 0
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            # try:
                gcn_bucket = []
                for item in bucket:
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
                    A_fw, A_bw,A_fw_dig,A_bw_dig, X = \
                        aux.create_graph_from_sentence_and_word_vectors(words,word_embeddings,subj_start,subj_end,
                                                                        obj_start,obj_end,maxlength)
                    # print(word_is_entity)
                    value_matrix_1 = aux.get_value_matrix(subj_start, subj_end, obj_start, obj_end, maxlength)
                    # print(value_matrix_1)
                    gcn_bucket.append((A_fw, A_bw, X, relation_vector ,value_matrix_1, A_fw_dig, A_bw_dig))
                    # print(gcn_bucket)
                # print(len(gcn_bucket))

                if len(gcn_bucket) >= 1:
                    loss = gcn_train.train(data = gcn_bucket, maxlength=maxlength)
                    # if cnt_batch % 1 == 0:
                    import time
                    now = time.strftime("%H:%M:%S")
                    print(now + " This step is Epoch: " + str(i) + ", Batch: " + str(cnt_batch) + ", The loss is: " + str(loss))
                    cnt_batch += 1
            # except:
            #     pass
        save_filename = saving_dir + '/gcn-re-' + str(i) + '.tf'
        sys.stderr.write('Saving into ' + save_filename + '\n')
        gcn_train.save(save_filename)
    return gcn_train
