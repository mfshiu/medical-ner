import os
import sys
import numpy as np
# import pandas as pd
# from tqdm import tqdm
from ckiptagger import construct_dictionary, WS
from common import util


NAME_ENTITY_MARK = "@N^E_M@"


def loadInputFile2(file_path):
    training_set = list()  # store trainingset [content,content,...]
    named_entities = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]

    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')

    articles = file_text.split('\n\n--------------------\n\n')[:-1]
    for article in articles:
        lines = article.split('\n')
        training_set.append(lines[0])     # original text

        nes = []
        for line in lines[2:]:
            # article_id, start_pos, end_pos, entity_text, entity_type
            annotations = line.split('\t')
            nes.append(annotations)
        named_entities.append(nes)

    return training_set, named_entities


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Error: need to specify source file name and output file name")
        print("Usage: python convert.py source_file output_file named_entities_file")
        sys.exit(-1)

    source_file = sys.argv[1]
    output_file = sys.argv[2]
    named_entities_file = sys.argv[3]

    default_name_entities = dict([(a, 'time') for a in util.get_time_entities()])
    articles, named_entities = loadInputFile2(source_file)

    for i, article in enumerate(articles):
        nes = [ne[3] for ne in named_entities[i]]
        for ne in nes:
            article = article.replace(ne, NAME_ENTITY_MARK)
        articles[i] = article

    print("Segmenting...", end=' ')
    ws = WS("./ckipdata")
    load_nes = util.load_name_entities("dataset/named_entities.txt")
    default_name_entities.update(load_nes)
    coerce_words = dict([(k, 1) for k in load_nes])
    coerce_words.update({NAME_ENTITY_MARK: 1})
    article_words = ws(articles, coerce_dictionary=util.construct_dictionary(coerce_words))
    print("done.")

    print("Geterating data...")
    train_data = []
    # for ss, ne in article_words, named_entities:
    for i, ss in enumerate(article_words):
        print("[%d] %s..."%(i, ss[:10]))
        for ne in named_entities[i]:
            k, v = ne[3], ne[4]
            train_data.append(k.ljust(29) + "_" + v + "\n")
            if k not in default_name_entities and not k.isdigit():
                default_name_entities[k] = v
        for s in ss:
            if NAME_ENTITY_MARK != s:
                if s in default_name_entities:
                    train_data.append(s.ljust(29) + "_" + default_name_entities[s] + "\n")
                else:
                    train_data.append(s.ljust(29) + "_" + "O" + "\n")
    train_data[-1] = train_data[-1][:-1]
    print("done.")

    print("Writing to output file...", end=' ')
    with open(output_file, 'w', encoding='utf8') as fp:
        fp.writelines(train_data)
    print("done.")

    print("Writing to name entities file...", end=' ')
    default_name_entities["個管師"] = "O"
    lines = [k + " " + v + "\n" for k, v in default_name_entities.items()]
    lines[-1] = lines[-1][:-1]
    with open(named_entities_file, 'w', encoding='utf8') as fp:
        fp.writelines(lines)
    print("done.")
