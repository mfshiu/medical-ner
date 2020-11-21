# coding: utf-8
import os
import sys
sys.path.append('..')
import numpy as np
from common.config import GPU
from common.config import Device as GPU_Device
from dataset import sequence
from attention_seq2seq import AttentionSeq2seq
from gen_input_data2 import convert_type_to_name
from gen_input_data3 import segment_data, load_recommend_dictionary

x_train, t_train = sequence.load_data_without_test('train_v4.txt', shuffle=False)
char_to_id, id_to_char = sequence.get_vocab()
vocab_size = len(char_to_id)

wordvec_size = 16
hidden_size = 256 * 2
batch_size = 128 * 2

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params("medical_v4-20.pkl")

test_file = "../dataset/validation.txt"
# test_file = "../dataset/development_2.txt"
fillout_file = "../dataset/aicup-output-v3-40.tsv"

question_size = 29
answer_size = 3
answer_none = "_O".ljust(answer_size)
window_size = 3

default_name_entities = load_recommend_dictionary()

# Read all articles
print("Read all articles...", end=' ')
articles = []
with open(test_file, "r") as fp:
    articles = [a for i, a in enumerate(fp) if (i - 1) % 5 == 0]
print("done.")

# Recognize name entities
print("Recognize name entities...")

# Write header
with open(fillout_file, "w") as fp:
    fp.write("article_id\tstart_position\tend_position\tentity_text\tentity_type\n")

def convert_to_word_id(sentences):
    questions = []
    for q in sentences:
        q = " ".join(q)
        q = q.strip()[:question_size]
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


def gen_sentences(article):
    words = segment_data([article])
    cnt = window_size - len(words) % window_size
    if cnt == window_size:
        cnt = 0
    words.extend(["無" for i in range(cnt)])

    sentences = []
    for i in range(0, len(words), window_size):
        sentences.append(words[i: i + window_size])

    return sentences


def guess_type(question, correct):
    correct = correct.flatten()
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))
    guess_text = ''.join([id_to_char[int(c)] for c in guess]).strip()

    return guess_text


# Recognize words in each article
mark = ['|','/','-','\\']
for article_id, article in enumerate(articles):
    start_position = 0
    print("\r\nRecognize article %d, %s ..."%(article_id, article[:50]))

    sentences = gen_sentences(article)
    rows = []
    x, t = convert_to_word_id(sentences)
    for i, words in enumerate(sentences):
        sentence = "".join(words)
        guess_text = guess_type(x[[i]], t[[i]])
        guess_chars = [c for c in guess_text]
        print("\r" + mark[i%4], end="")
        print("[%d-%d] %s => %s" % (article_id, i, sentence, guess_text), end="")

        offset = 0
        for j, g in enumerate(guess_chars):
            w = words[j]
            if g != "O":
                type_name = convert_type_to_name(g)
                row = "{}\t{}\t{}\t{}\t{}\n".format(
                    article_id, start_position + offset, start_position + offset + len(w), w, type_name)
                rows.append(row)
                print("\r[%d] %s" % (i, row), end="")
            offset += w

        start_position += len(sentence) # + 1

    with open(fillout_file, 'a') as fp:
        fp.writelines(rows)

print("Recognize name entities, done.")
