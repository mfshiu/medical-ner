import sys


max_question_length = 1000
separator = ' '


def generate_out_line(question, answer):
    q = separator.join(question)
    if len(q) > max_question_length:
        print("\tExceed max question length %d: %s" % (max_question_length, q))
        return None
    q = q.ljust(max_question_length)

    a = separator.join(answer)
    out_line = q + "_" + a

    return out_line


def convert(source_file, target_file):
    output_lines = []

    with open(source_file, 'r') as fp:
        question, answer, cnt = [], [], 0
        for in_line in fp:
            cnt += 1
            in_line = in_line.strip()
            print("\rLine: [%d] %s" % (cnt, in_line))
            if len(in_line) == 0:
                out_line = generate_out_line(question, answer)
                if out_line is not None:
                    output_lines.append(out_line)
                    print("\n[%d] %s" % (len(output_lines), out_line))
                question, answer = [], []
            else:
                tokens = in_line.split(' ', 1)
                if len(tokens) == 2:
                    question.append(tokens[0].strip())
                    answer.append(tokens[1].strip())
                else:
                    print("\nWrong source line format: [%d] %s" % (cnt, in_line, ))

    print("\nWriting to: %s", (target_file,))
    with open(target_file, 'w') as fp:
        fp.writelines(output_lines)
    print("Done.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: need to specify source file name and target file name")
        print("Usage: python convert_ner.py source_file target_file")
        sys.exit(-1)

    source_file = sys.argv[1]
    target_file = sys.argv[2]
    convert(source_file, target_file)
