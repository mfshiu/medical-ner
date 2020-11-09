# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from dataset import sequence
from attention_seq2seq import AttentionSeq2seq

x_train, t_train = sequence.load_data_without_test('train_33839.txt', shuffle=False)
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
model.load_params("medical-16.pkl")

test_file = "../dataset/test-medical.txt"
fillout_file = "../dataset/test-medicalout.txt"
out_lines = []

questions = []
with open(test_file) as fp:
    for i, q in enumerate(fp):
        q = q.strip()
        if len(q) > 29:
            q = q[:29]
        questions.append(q.ljust(29)[::-1])

ans_none = "_O".ljust(15)
x = np.zeros((len(questions), len(questions[0])), dtype=np.int)
t = np.zeros((len(questions), len(ans_none)), dtype=np.int)


dummy_id = char_to_id["_"]
def get_char_id(c):
    id = dummy_id
    if c in char_to_id:
        id = char_to_id[c]
    return id


for i, sentence in enumerate(questions):
    x[i] = [get_char_id(c) for c in list(sentence)]
    t[i] = [get_char_id(c) for c in list(ans_none)]

total = len(x)
for i in range(total):
    question, correct = x[[i]], t[[i]]

    correct = correct.flatten()
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 轉換成字串
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    guess = ''.join([id_to_char[int(c)] for c in guess])
    question = question.strip()[::-1]
    if len(question) > 0:
        out_line = question.strip() + " " + guess.strip()
        out_lines.append(out_line + "\n")
        print("[%d] %s"%(i, out_line))
    else:
        out_lines.append("\n")
        print("")


with open(fillout_file, 'w') as fp:
    fp.writelines(out_lines)
