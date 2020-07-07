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
    import numpy as np
    import GCN_RE.utils.auxiliary as aux

    import GCN_RE
    print("---dataset is ready, now we are taking them from file---")
    sentences = aux.get_all_sentence(dataset, data_name)
    maxlength = 256
    print('---get the embeddings of sentence---')
    data = aux.get_data_from_sentences(sentences, maxlength)
    print("---make the batch and get the adj matrix---")
    buckets = bin_data_into_buckets(data, bucket_size)
    gcn_model = GCN_RE.GCN_model.Train_and_E(maxlength=maxlength)

    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        cnt_batch = 0
        correct_num = 0
        all_num = 0
        all_batch = len(random_buckets)
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            # try:
            loss = 0
            gcn_bucket = []
            for item in bucket:
                '''
                word_list, word_vector, subj_start, subj_end, obj_start,obj_end, relation_vector, edges
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
                edges = item[7]

                A_fw, A_bw, X = \
                    aux.create_graph_from_sentence_and_word_vectors(words, word_embeddings, subj_start, subj_end,
                                                                    obj_start, obj_end, edges)
                # print(word_is_entity)
                # print(value_matrix_1)
                gcn_bucket.append((A_fw, A_bw, X, relation_vector,subj_start,subj_end, obj_start, obj_end))
                # print(gcn_bucket)
            # print(len(gcn_bucket))
            correct = 0
            if len(gcn_bucket) >= 1:
                correct, loss = gcn_model.train(data = gcn_bucket)
                # if cnt_batch % 1 == 0:
            import time
            now = time.strftime("%H:%M:%S")
            correct_num += correct
            all_num += len(gcn_bucket)
            # print(now + " This step is Epoch: " + str(i) + ", Batch: " + str(cnt_batch) + ", The loss is: " + str(loss.data[0])
            #       + ", acc is: " + str(acc.data[0]))
            cnt_batch += 1
            if cnt_batch%10 == 0 or cnt_batch == all_batch:
                print('{}, This Step is Epoch {}, Batch {}/{}, loss: {:.4f}, acc: {:.2f}'.
                      format(now, i, cnt_batch, all_batch, loss, correct_num/all_num))
                correct_num = 0
                all_num = 0
            # except:
            #     pass
        save_filename = saving_dir + '/gcn-re-param-' + str(i) + '.pkl'
        sys.stderr.write('Saving into ' + save_filename + '\n')
        gcn_model.save(save_filename)
    return gcn_model
