import json

with open('./test_mini.json', 'r', encoding='utf-8') as data_input:
    sentences = []
    lines = json.load(data_input)
    print(len(lines))
    # print(len(lines['sentence']))
    maxlength = 0
    for line in lines:
        sentence = line['sentence']
        subj_start = line['subj_start']
        subj_end = line['subj_end']
        obj_start = line['obj_start']
        obj_end = line['obj_end']
        relation = line['relation']

        # print(relation_vector)
        if len(sentence) > maxlength:
            maxlength = len(sentence)

    print(maxlength)