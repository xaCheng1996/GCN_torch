import spacy
import re
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

parser = spacy.load('en_core_web_md')

_pad_word = 'word'

default_vector = np.array(parser(_pad_word)[0].vector, dtype=np.float64)

tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
        "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
        "WP$", "WRB", '``', "''", '.', ',', ':', '-LRB-', '-RRB-']

classes = ["PERSON", "ORGANIZATION", "LOCATION"]

relation_TacRED = ['org:founded_by', 'no_relation', 'per:employee_of', 'per:cities_of_residence',
                    'per:children', 'per:title', 'per:siblings', 'per:religion', 'org:alternate_names',
                    'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees',
                    'per:countries_of_residence', 'org:city_of_headquarters', 'org:members',
                    'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters',
                    'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin',
                    'org:political/religious_affiliation', 'per:age', 'per:other_family',
                    'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders',
                    'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death',
                    'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth',
                    'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death'
                   ]

word_substitutions = {'-LRB-': '(',
                      '-RRB-': ')',
                      '``': '"',
                      "''": '"',
                      "--": '-',
                      }



# just clean the sentence

def create_full_sentence(words):
    import re

    sentence = ' '.join(words)
    sentence = re.sub(r' (\'[a-zA-Z])', r'\1', sentence)
    sentence = re.sub(r' \'([0-9])', r' \1', sentence)
    sentence = re.sub(r' (,.)', r'\1', sentence)
    sentence = re.sub(r' " (.*) " ', r' "\1" ', sentence)
    sentence = sentence.replace('do n\'t', 'don\'t')
    sentence = sentence.replace('did n\'t', 'didn\'t')
    sentence = sentence.replace('was n\'t', 'wasn\'t')
    sentence = sentence.replace('were n\'t', 'weren\'t')
    sentence = sentence.replace('is n\'t', 'isn\'t')
    sentence = sentence.replace('are n\'t', 'aren\'t')
    sentence = sentence.replace('\' em', '\'em')
    sentence = sentence.replace('s \' ', 's \'s ')
    return sentence


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False

# get vector from word with Spacy

def get_clean_word_vector(word):
    if word in word_substitutions:
        word = word_substitutions[word]
    word_vector = np.array(parser.vocab[word].vector, dtype=np.float64)
    if vector_is_empty(word_vector):
        word_vector = default_vector
    return word_vector

def get_relation_vector(relation, relation_class):
    relation_vector = [0] * len(relation_class)
    try:
        index = relation_class.index(relation)
    except:
        index = 1
    relation_vector[index] = 100
    return relation_vector

def create_graph_from_sentence_and_word_vectors(words, word_vectors, subj_start, subj_end, obj_start, obj_end, edges):
    X = []
    matrix_len = len(words)
    # print(words)
    # print(matrix_len)
    A_fw = np.zeros(shape=(matrix_len, matrix_len))
    A_bw = np.zeros(shape=(matrix_len, matrix_len))

    for i in range(matrix_len):
        A_fw[i][i] = 1
        A_bw[i][i] = 1

    X.append(word_vectors)

    for (word1,word2) in edges:
        if word1 >= matrix_len or word2 >= matrix_len:
            continue
        else:
            A_fw[word1][word2] += 1
            A_fw[word2][word1] += 1


    for i in range(subj_start, subj_end + 1):
        for j in range(subj_start, subj_end + 1):
            A_bw[i][i] += 1
            A_bw[i][j] += 1
            A_bw[j][i] += 1
            A_bw[j][j] += 1

    for i in range(obj_start, obj_end+1):
        for j in range(obj_start, obj_end + 1):
            A_bw[i][i] += 1
            A_bw[i][j] += 1
            A_bw[j][i] += 1
            A_bw[j][j] += 1

    return A_fw, A_bw, X


def get_all_sentence(filename, data_name):
    if data_name == 'TacRED':
        maxlength = 256
        with open(filename, 'r', encoding='utf-8') as data_input:
            sentences = []
            lines = json.load(data_input)
            for line in lines:
                sentence = line['sentence']
                subj_start = line['subj_start']
                subj_end = line['subj_end']
                obj_start = line['obj_start']
                obj_end = line['obj_end']
                relation = line['relation']
                edges = line['edges']

                relation_vector = get_relation_vector(relation, relation_TacRED)
                # print(relation_vector)
                sentence = list(sentence)

                sentences.append([sentence, subj_start, subj_end, obj_start, obj_end, relation_vector, edges])
            return sentences
    else:
        print("no dataset! ")


def get_data_from_sentences(sentences, maxlength):
    all_data = []
    cnt = 0
    for sentence in sentences:
        word_vector = []
        word_list = sentence[0]
        subj_start = sentence[1]
        subj_end = sentence[2]
        obj_start = sentence[3]
        obj_end = sentence[4]
        relation_vector = sentence[5]
        edges = sentence[6]

        # print(maxlength)
        for word in word_list:
            word_vector.append(get_clean_word_vector(word))

        # if len(word_list) < maxlength:
        #     for i in range(maxlength - len(word_list)):
        #         word_list.append(_pad_word)
        #
        # if len(word_vector) < maxlength:
        #     for i in range(maxlength-len(word_vector)):
        #         word_vector.append(default_vector)

        if cnt % 5000 == 0:
            print("Have processed data: " + str(cnt))
        cnt += 1

        all_data.append((word_list, word_vector, subj_start, subj_end, obj_start,obj_end, relation_vector, edges))

    return all_data
