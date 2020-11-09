# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from dataset import sequence
from attention_seq2seq import AttentionSeq2seq
from ckiptagger import construct_dictionary, WS
from common import util


x_train, t_train = sequence.load_data_without_test('train_180000.txt', shuffle=False)
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
model.load_params("medical-37.pkl")

test_file = "../dataset/development_1.txt"
fillout_file = "../output/development_1-out.txt"

question_size = 29
answer_size = 15
answer_none = "_O".ljust(answer_size)

# Read all articles
print("Read all articles...", end=' ')
articles = []
with open(test_file, "r") as fp:
    articles = [a for i, a in enumerate(fp) if (i - 1) % 5 == 0]
print("done.")

# Segment articles
# generate coerce dictionary
def gen_coerec_dic():
    dic = {}
    return util.construct_dictionary(dic)

article_words = []
print("Segment articles...", end=' ')
ws = WS("../ckipdata")
article_words = ws(articles, coerce_dictionary=gen_coerec_dic())
print("done.")

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


# Recognize words in each article
ans_none = answer_none[1:]
for article_id, words in enumerate(article_words):
    print("Recognize article %d, %s ..."%(article_id, articles[article_id][:50]))

    start_position, end_position = 0, 0
    rows = []
    x, t = convert_to_word_id(words)
    for i in range(len(x)):
        question, correct = x[[i]], t[[i]]
        correct = correct.flatten()
        start_id = correct[0]
        correct = correct[1:]
        guess = model.generate(question, start_id, len(correct))
        guess_text = ''.join([id_to_char[int(c)] for c in guess])
        word = words[i]
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
