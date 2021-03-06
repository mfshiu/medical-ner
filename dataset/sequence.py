# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy


id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name, seed=1984):
    que, ans = load_data_without_test(file_name, seed)

    # 10% for validation set
    train_size = len(que) - int(min(1000, len(que) / 10))
    (que_train, que_test) = que[:train_size], que[train_size:]
    (ans_train, ans_test) = ans[:train_size], ans[train_size:]

    return (que_train, ans_train), (que_test, ans_test)


def load_data2(file_name='addition.txt', seed=1934):
    que, ans = load_data_without_test(file_name, seed)
    ans_none = char_to_id["O"]

    (que_train, que_test) = ([], [])
    (ans_train, ans_test) = ([], [])
    for i, q in enumerate(que):
        a = ans[i]
        if len(que_test) >= 1000:
            que_train.append(q)
            ans_train.append(a)
        elif a[1] == ans_none and numpy.random.randint(0, 99) == 0:
            que_test.append(q)
            ans_test.append(a)
        else:
            que_train.append(q)
            ans_train.append(a)
    return (que_train, ans_train), (que_test, ans_test)


question_size = 29
answer_size = 30


def load_data_without_test(file_name, shuffle=True):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, 'r'):
        idx = line.find('_')
        questions.append(line[:idx])
        answers.append(line[idx:-1].ljust(answer_size))

    # create vocab dict
    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)

    # create numpy array
    x = numpy.zeros((len(questions), question_size), dtype=numpy.int)
    t = numpy.zeros((len(questions), answer_size), dtype=numpy.int)

    for i, sentence in enumerate(questions):
        print("\r[%d] %s"%(i, sentence), end="")
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        print("\r[%d] %s"%(i, sentence), end="")
        t[i] = [char_to_id[c] for c in list(sentence)]
    print()

    if shuffle:
        indices = numpy.arange(len(x))
        numpy.random.seed(1972)
        numpy.random.shuffle(indices)
        x = x[indices]
        t = t[indices]

    return x, t


def load_word_id(file_name):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions = []

    for line in open(file_path, 'r'):
        line = line.ljust(29)
        questions.append(line)

    x = numpy.zeros((len(questions), len(questions[0])), dtype=numpy.int)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]

    return x


def get_vocab():
    return char_to_id, id_to_char
