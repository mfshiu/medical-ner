import os
import sys
import numpy as np
# import pandas as pd
# from tqdm import tqdm
from ckiptagger import construct_dictionary, WS


NAME_ENTITY_MARK = "@N^E_M@"


def construct_dictionary(word_to_weight):
    length_word_weight = {}

    for word, weight in word_to_weight.items():
        if not word: continue
        try:
            weight = float(weight)
        except ValueError:
            continue
        length = len(word)
        if length not in length_word_weight:
            length_word_weight[length] = {}
        length_word_weight[length][word] = weight

    length_word_weight = sorted(length_word_weight.items())

    return length_word_weight


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
    if len(sys.argv) != 3:
        print("Error: need to specify source file name and output file name")
        print("Usage: python convert.py source_file output_file")
        sys.exit(-1)

    source_file = sys.argv[1]
    output_file = sys.argv[2]
    articles, named_entities = loadInputFile2(source_file)

    for i, article in enumerate(articles):
        nes = [ne[3] for ne in named_entities[i]]
        for ne in nes:
            article = article.replace(ne, NAME_ENTITY_MARK)
        articles[i] = article

    print("Segmenting...", end=' ')
    ws = WS("./ckipdata")
    article_words = ws(articles, coerce_dictionary=construct_dictionary({
        NAME_ENTITY_MARK: 1,
    }))
    print("done.")

    print("Geterating data...")
    train_data = []
    # for ss, ne in article_words, named_entities:
    for i, ss in enumerate(article_words):
        print("[%d] %s..."%(i, ss[:10]))
        for ne in named_entities[i]:
            # train_data.append(ne[3], ne[4])
            train_data.append(ne[3].ljust(29) + "_" + ne[4] + "\n")
        for s in ss:
            if NAME_ENTITY_MARK != s:
                # train_data.append((s, "O"))
                train_data.append(s.ljust(29) + "_" + "O" + "\n")
    train_data[-1] = train_data[-1][:-1]
    print("done.")

    print("Writing to file...", end=' ')
    with open(output_file, 'w', encoding='utf8') as fp:
        fp.writelines(train_data)
    print("done.")
