import spacy

parser = spacy.load('en_core_web_md')

_invalid_words = [' ']


class SpacyTagger:

    def __init__(self, sentence):
        self.sentence = sentence


class SpacyParser:

    def __init__(self, tagger):
        self.tagger = tagger
        self.parser = parser

    def execute(self):
        parsed = self.parser(self.tagger.sentence)
        edges = []
        names = []
        words = []
        tags = []
        types = []

        i = 0
        items_dict = dict()
        for item in parsed:
            # for l in item.children:
            #     print("test " + str(l.orth_))
            if item.orth_ in _invalid_words:
                continue
            items_dict[item.idx] = i
            i += 1

        for item in parsed:
            if item.orth_ in _invalid_words:
                continue
            index = items_dict[item.idx]
            for child_index in [items_dict[l.idx] for l in item.children
                                if not l.orth_ in _invalid_words]:
                edges.append((index, child_index))
            names.append(index)
            words.append(item.vector)
            tags.append(item.tag_)
            types.append(item.dep_)

        return names, edges, words, tags, types


    def execute_layer2(self, subj_start, subj_end, obj_start, obj_end):
        edges = []
        # print(self.tagger.word_isEntity)
        entity_dict = dict()
        for index in range(subj_start, subj_end + 1):
            entity_dict[index] = 1

        for index in range(obj_start, obj_end + 1):
            entity_dict[index] = 1

        # for i in entity_dict:
        #     print(str(i)+" "+str(entity_dict[i]))
        for key_i in entity_dict.keys():
            for key_j in entity_dict.keys():
                edges.append((key_i,key_j))

        # print(names)
        return edges