﻿# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *


delimiters = "：，。？；！.,;!?"


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

# generate coerce dictionary
def gen_word_to_weight(addition_dic={}):
    tes = get_time_entities()

    word_to_weight = {}
    for te in tes:
        word_to_weight[te] = 1

    specials = ['個管師']
    for s in specials:
        word_to_weight[s] = 1

    word_to_weight.update(addition_dic)
    return word_to_weight


# def get_conunt_entities():
#     name_entities = []
#     mon1 = [i for i in range(1,13)]
#     units = ["個", "顆", "次", "下", "串", "包", "張", "袋", "本", "罐", "箱", "項"
#         , "項", "項", "項", "項", "項", "項", "項", "項", "項"]
#     days = [31 ,30, 29 ,30 ,31 ,30 ,31 ,31 ,30 ,31 ,30 ,31]
#     day_name = ['', '日', '號']
#     word_to_weight = {}
#
#     for mm in [mon1, mon2]:
#         for i, m in enumerate(mm):
#             for d in range(days[i]):
#                 for dn in day_name:
#                     time_entities.append("{}月{}{}".format(m, d+1, dn))
#             for d in mon2[:10]:
#                 for dn in day_name:
#                     time_entities.append("{}月{}{}".format(m, d, dn))
#             for d in mon2[:9]:
#                 for dn in day_name:
#                     time_entities.append("{}月十{}{}".format(m, d, dn))
#                     time_entities.append("{}月二十{}{}".format(m, d, dn))
#             for dn in day_name:
#                 time_entities.append("{}月三十{}".format(m, dn))
#                 time_entities.append("{}月三十一{}".format(m, dn))
#
#     day_name = ['日', '號', '天']
#     for d in range(31):
#         for dn in day_name:
#             word_to_weight["{}{}".format(d + 1, dn)] = 1
#
#     units = ['分', '分鐘']  #, '個', '次']
#     num = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "兩"]
#     for u in units:
#         for m in range(60):
#             time_entities.append("{}{}".format(m + 1, u))
#         for m in num:
#             time_entities.append("{}{}".format(m, u))
#
#     return time_entities


def get_time_entities():
    time_entities = []
    mon1 = [i for i in range(1,13)]
    mon2 = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
    days = [31 ,30, 29 ,30 ,31 ,30 ,31 ,31 ,30 ,31 ,30 ,31]
    day_name = ['', '日', '號']
    word_to_weight = {}

    for mm in [mon1, mon2]:
        for i, m in enumerate(mm):
            for d in range(days[i]):
                for dn in day_name:
                    time_entities.append("{}月{}{}".format(m, d+1, dn))
            for d in mon2[:10]:
                for dn in day_name:
                    time_entities.append("{}月{}{}".format(m, d, dn))
            for d in mon2[:9]:
                for dn in day_name:
                    time_entities.append("{}月十{}{}".format(m, d, dn))
                    time_entities.append("{}月二十{}{}".format(m, d, dn))
            for dn in day_name:
                time_entities.append("{}月三十{}".format(m, dn))
                time_entities.append("{}月三十一{}".format(m, dn))

    day_name = ['日', '號', '天']
    for d in range(31):
        for dn in day_name:
            word_to_weight["{}{}".format(d + 1, dn)] = 1

    units = ['分', '分鐘']  #, '個', '次']
    num = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "兩"]
    for u in units:
        for m in range(60):
            time_entities.append("{}{}".format(m + 1, u))
        for m in num:
            time_entities.append("{}{}".format(m, u))

    return time_entities


def load_name_entities(file_path):
    with open(file_path, 'r', encoding='utf8') as fp:
        lines = fp.readlines()

    nes = {}
    for line in lines:
        tokens = line.split(" ")
        k, v = tokens[0].strip(), tokens[1].strip()
        nes[k] = v

    return nes


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    '''計算餘弦相似度

    :param x: 向量
    :param y: 向量
    :param eps: 防止”除以0”的小數值
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''相似詞搜尋

    :param query: 查詢（文本）
    :param word_to_id: 將字詞轉換成字詞ID的字典
    :param id_to_word: 將字詞ID轉換成字詞的字典
    :param word_matrix: 整合詞向量的矩陣。用來儲存對應各列的詞向量    :param top: 要顯示到第幾名為止
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    '''轉換成one-hot編碼

    :param corpus: 字詞ID清單（一維或二維NumPy陣列）
    :param vocab_size: 語彙量
    :return: one-hot編碼（二維或三為NumPy陣列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''建立共生矩陣

    :param corpus: 語料庫（字詞ID清單）
    :param vocab_size:語彙量
    :param window_size:視窗大小（視窗大小為1時，字詞左右各1個字為上下文）
    :return: 共生矩陣
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps = 1e-8):
    '''建立PPMI（下一個正向點間互資訊）

    :param C: 共生矩陣
    :param verbose: 是否輸出執行狀況    
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M


def create_contexts_target(corpus, window_size=1):
    '''轉換成one-hot編碼

    :param words: 字詞ID的NumPy陣列
    :param vocab_size: 語彙量
    :return: 轉換成one-hot編碼後的NumPy陣列
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 開頭的分隔字元
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 轉換成字串
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    is_right = correct == guess
    if verbos or not is_right:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q:', question, end = ', ')
        print('T:', correct, end = ', ')

        is_windows = os.name == 'nt'

        if is_right:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        sys.stdout.flush()
        # print('---')

    return 1 if guess == correct else 0


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
