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


def convert_name_to_type(et):
    if len(et) > 2:
        name = et[2:]
        if name in ets:
            if et[0] == "B":
                return ets[name]
            else:
                return ets[name].lower()
        else:
            return empty_sign
    else:
        return empty_sign


def load_coerce_dictionary(custom_dict=None):
    file_path = root_dir + '/dataset/dict_coerce.txt'
    coerce_dict = util.load_dictionary(file_path)
    if custom_dict:
        coerce_dict.update(custom_dict)

    return coerce_dict


global _recommend_dict
_recommend_dict = None

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
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            annot = annot.split('\t')  # annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions


def prepare_data(trainingset, mentions):
    custom_dict = load_recommend_dictionary(mentions)
    coerce_dict = load_coerce_dictionary()
    custom_dict.update(coerce_dict)
    words = segment_data(trainingset, mentions)
    trains = []
    for word in words:
        if word in custom_dict:
            entity_type = ets[custom_dict[word]]
        else:
            entity_type = "O"
        trains.append((word, (entity_type * len(word)).capitalize()))

    train_data = []
    for i in range(0, len(trains)-5, 2):
        train_data.append(("".join([x[0] for x in trains[i: i+5]]),
                           "".join([x[1] for x in trains[i: i+5]])))

    return train_data


def segment_data(articles, mentions=None):
    recommend_words = dict([(k, 1) for k in load_recommend_dictionary(mentions)])
    coerce_words = dict([(k, 1) for k in load_coerce_dictionary()])
    print("Segment all articles...", end=' ')
    ws = WS(root_dir + "/ckipdata")
    # delimiters = set([char for char in "：，。？；！"])
    atricle_words = ws(articles,
                       # sentence_segmentation=True,
                       # segment_delimiter_set=delimiters,
                       recommend_dictionary=util.construct_dictionary(recommend_words),
                       coerce_dictionary=util.construct_dictionary(coerce_words))
    del ws
    words = []
    for sublist in atricle_words:
        words.extend(sublist)
    # words = list(filter(lambda w: w not in delimiters, words))
    print("done.")
    print("Total %d words."%(len(words),))

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

    trainingset, position, mentions = loadInputFile(input_path)
    train_data = prepare_data(trainingset, mentions)
    write_down(train_data, output_path)
