import os
import sys
from common import util
from ckiptagger import construct_dictionary, WS

question_size = 29
empty_sign = "O"
ets = {
    "account": "A",
    "biomarker": "B",
    "clinical_event": "C",
    "contact": "D",
    "education": "E",
    "family": "F",
    "belonging_mark": "G",
    "others": "H",
    "ID": "I",
    "location": "L",
    "money": "M",
    "name": "N",
    "none": "O",
    "organization": "R",
    "profession": "P",
    "special_skills": "S",
    "time": "T",
    "unique_treatment": "U",
    "med_exam": "X",
}
root_dir = os.path.dirname(os.path.abspath(__file__))
global _recommend_dict
_recommend_dict = None


def load_coerce_dictionary(custom_dict=None):
    file_path = root_dir + '/dataset/dict_coerce.txt'
    coerce_dict = util.load_dictionary(file_path)
    if custom_dict:
        coerce_dict.update(custom_dict)

    return coerce_dict


def load_recommend_dictionary(custom_dict=None):
    global _recommend_dict

    if _recommend_dict:
        return _recommend_dict

    file_path = root_dir + '/dataset/dict_recommend.txt'
    _recommend_dict = util.load_dictionary(file_path)

    time_dict = dict([(a, 'time') for a in util.get_time_entities()])
    _recommend_dict.update(time_dict)

    if custom_dict:
        _recommend_dict.update(custom_dict)

    return _recommend_dict


def loadInputFile(file_path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = list()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        posi = list()
        ment = dict()
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            annot = annot.split('\t')  # annot= article_id, start_pos, end_pos, entity_text, entity_type
            posi.append(tuple(annot))
            ment[annot[3]] = annot[4]

        position.append(posi)
        mentions.append(ment)

    return trainingset, position, mentions


def prepare_data(articles, positions, mentions):
    train_data = []

    recommend_dict = load_recommend_dictionary()
    coerce_dict = load_coerce_dictionary()
    for article_id, article in enumerate(articles):
        position = positions[article_id]
        custom_dict = mentions[article_id]
        custom_dict.update(coerce_dict)
        print("[%d/%d] "%(article_id, len(articles)), end="")
        words = segment_data([article], recommend_dict, custom_dict)

        known_dict = {}
        known_dict.update(recommend_dict)
        known_dict.update(coerce_dict)
        types = []
        word_pos = 0
        pos = position.pop(0)
        for w in words:
            if str(word_pos) == pos[1]:
                types.append(ets[pos[4]])
                if position:
                    pos = position.pop(0)
            elif w.lower() in known_dict:
                types.append(ets[known_dict[w.lower()]])
            else:
                types.append("O")
            word_pos += len(w)

        for i in range(0, len(words)-3):
            train_data.append((" ".join([x for x in words[i: i+3]]),
                               "".join([x for x in types[i: i+3]])))

    return train_data


ws = WS("./ckipdata")
def segment_data(articles, recommend_dict, coerce_dict):
    recommend_words = dict([(k, 1) for k in recommend_dict])
    coerce_words = dict([(k, 1) for k in coerce_dict])
    print("Segment: %s..."%(articles[0][:50]))
    delimiters = set([char for char in "：，。？；！"])
    atricle_words = ws(articles,
                       sentence_segmentation=True,
                       segment_delimiter_set=delimiters,
                       recommend_dictionary=util.construct_dictionary(recommend_words),
                       coerce_dictionary=util.construct_dictionary(coerce_words))
    words = []
    for sublist in atricle_words:
        words.extend(sublist)
    print("Total %d words: %s｜..."%(len(words), "｜".join(words[:20])))

    return words


def write_down(train_data, output_path):
    output_lines = []
    for data in train_data:
        q, a = data[0], data[1]
        line = q[:question_size].ljust(question_size) + "_" + a[:question_size]
        output_lines.append(line + "\n")

    with open(output_path, "w") as fp:
        fp.writelines(output_lines)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: need to specify source file name and output file name")
        print("Usage: python convert.py source_file output_file named_entities_file")
        sys.exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    articles, positions, mentions = loadInputFile(input_path)
    train_data = prepare_data(articles, positions, mentions)
    write_down(train_data, output_path)
    print("Completed.")
