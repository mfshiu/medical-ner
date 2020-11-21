# coding: utf-8
import os
import sys
sys.path.append('..')
import numpy as np
from common.config import GPU
from common.config import Device as GPU_Device
from dataset import sequence
from attention_seq2seq import AttentionSeq2seq
from ckiptagger import construct_dictionary, WS, POS, NER
from common import util

# if GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_Device)

x_train, t_train = sequence.load_data_without_test('train_300000.txt', shuffle=False)
char_to_id, id_to_char = sequence.get_vocab()
vocab_size = len(char_to_id)

# x_test, t_test = sequence.load_data_without_test('test-sample.txt', shuffle=False)
#
# # 反轉輸入內容
# x_test = x_test[:, ::-1]

# 設定超參數
# vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256 * 2
batch_size = 128 * 2

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params("medical-60.pkl")

test_file = "../dataset/development_2.txt"
fillout_file = "../output/development_2-out.tsv"

question_size = 29
answer_size = 15
answer_none = "_O".ljust(answer_size)

# Read all articles
print("Read all articles...", end=' ')
articles = []
with open(test_file, "r") as fp:
    articles = [a for i, a in enumerate(fp) if (i - 1) % 5 == 0]
print("done.")


ckip_type_map = {
    "GPE": "location", "ORG": "organization", "ORG": "organization", "DATE": "time", "PERSON": "name"
    , "TIME": "time", "FAC": "location", "LOC": "location", "d": "d", "e": "5"
}
ckip_dropped_words = set({'齁', '阿', '哈哈', '以', '恩恩'})
def parse_nes(ckip_entity_words):
    nes = {}
    for entity_words in ckip_entity_words:
        for entity in entity_words:
            word, word_type = entity[3], entity[2]
            if word_type in ckip_type_map \
                    and word not in ckip_dropped_words \
                    and not word.startswith('阿') \
                    and not word.startswith('恩') \
                    and not word.startswith('齁') \
                    and not word.startswith('哈'):
                nes[word] = ckip_type_map[word_type]
    return nes

# Segment articles
print("Segment articles...")
load_nes = util.load_dictionary("../dataset/named_entities.txt")
coerce_words = dict([(k, 1) for k in load_nes])
ws = WS("../ckipdata") # , disable_cuda=not GPU)
print("  CKIP Pos articles...")
pos = POS("../ckipdata")
print("  CKIP Ner articles...")
ner = NER("../ckipdata")
article_words = ws(articles, coerce_dictionary=util.construct_dictionary(coerce_words))
ckip_pos_words = pos(article_words)
ckip_entity_words = ner(article_words, ckip_pos_words)
ckip_nes = parse_nes(ckip_entity_words)
print("Segment articles done.")

# Recognize name entities
print("Recognize name entities...")

# Write header
with open(fillout_file, "w") as fp:
    fp.write("article_id\tstart_position\tend_position\tentity_text\tentity_type\n")

def convert_to_word_id(words):
    questions = []
    for i, q in enumerate(words):
        q = q.strip()
        if len(q) > question_size:
            q = q[:question_size]
        questions.append(q.ljust(question_size)[::-1])

    x = np.zeros((len(questions), len(questions[0])), dtype=np.int)
    t = np.zeros((len(questions), len(answer_none)), dtype=np.int)

    dummy_id = char_to_id["_"]
    def get_char_id(c):
        id = dummy_id
        if c in char_to_id:
            id = char_to_id[c]
        return id

    for i, sentence in enumerate(questions):
        x[i] = [get_char_id(c) for c in list(sentence)]
        t[i] = [get_char_id(c) for c in list(answer_none)]

    return x, t


def guess_type(question, correct):
    correct = correct.flatten()
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))
    guess_text = ''.join([id_to_char[int(c)] for c in guess]).strip()

    return guess_text


# Recognize words in each article
ans_none = answer_none[1:].strip()
for article_id, words in enumerate(article_words):
    print("Recognize article %d, %s ..."%(article_id, articles[article_id][:50]))

    start_position, end_position = 0, 0
    rows = []
    x, t = convert_to_word_id(words)
    for i in range(len(x)):
        word = words[i]
        if word in load_nes:
            guess_text = load_nes[word]
        elif word in ckip_nes:
            guess_text = ckip_nes[word]
        else:
            guess_text = guess_type(x[[i]], t[[i]])

        end_position = start_position + len(word)
        if ans_none != guess_text:
            row = "{}\t{}\t{}\t{}\t{}\n".format(
                article_id, start_position, end_position, word, guess_text)
            rows.append(row)
            print("[%d] %s" % (i, row), end="")
        start_position = end_position

    with open(fillout_file, 'a') as fp:
        fp.writelines(rows)

print("Recognize name entities, done.")
