import sys


max_question_length = 29
max_answer_length = 6
separator = ' '


def generate_out_line(question, answer):
    if len(question) > max_question_length:
        question = question[: max_question_length]
        print("\nExceed max question length %d, cut to: %s\n" % (max_question_length, question))

    question = question.ljust(max_question_length)
    answer = answer.ljust(max_answer_length)
    out_line = question + "_" + answer

    return out_line


def convert(source_file, target_file):
    output_lines = []

    with open(source_file, 'r') as fp:
        cnt = 0
        for in_line in fp:
            cnt += 1
            in_line = in_line.strip()
            print("\r[%d] %s" % (cnt, in_line), end="")
            if len(in_line) > 0:
                tokens = in_line.split(' ', 1)
                if len(tokens) == 2:
                    q = tokens[0].strip()
                    a = tokens[1].strip()
                    out_line = generate_out_line(q, a)
                    output_lines.append(out_line + "\n")
                else:
                    print("\nWrong source line format: [%d] %s" % (cnt, in_line, ))
            # else:
            #     output_lines.append("\n")

    print("\nWriting to: %s ..." % (target_file,), end=" ")
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
