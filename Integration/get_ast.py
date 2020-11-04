import javalang
import json
from tqdm import tqdm    # tqdmを使用して, プログレスバーを導入．ange()関数の値をtqdm()関数に与えるだけでプログレスバーを表示出来る．
import collections
import sys
import pprint


# ファイルを書き出すところ
path_w = 'C:\\Users\\acmil\\Desktop\\Fukasawa_mthesis\\EMSE-DeepCom\\data_utils\\Integration\\hogehoge.txt'

# 解析対象ソースコードに対する処理
# javaの解析対象（ソースコード）からsouce.codeを生成
def process_source(file_name, save_file):
    with open(file_name, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(save_file, 'w+', encoding='utf-8') as save:
        for line in lines:
            code = line.strip()
            tokens = list(javalang.tokenizer.tokenize(code))  # javalangを使って構文解析しトークンを抽出
            tks = []
            # クラス名を型で正規化
            for tk in tokens:
                if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                    tks.append('STR_')
                elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                    tks.append('NUM_')
                elif tk.__class__.__name__ == 'Boolean':
                    tks.append('BOOL_')
                else:
                    tks.append(tk.value)
            save.write(" ".join(tks) + '\n')


def get_ast(file_name, w):
    with open(file_name, 'r', encoding='utf-8') as f:
        original_code = f.read()
        # print(type(original_code))
        # print(original_code)
    # with open(w, 'w+', encoding='utf-8') as wf:
    with open(path_w, 'w', encoding='utf-8') as wf:
        code = original_code.split()
        # print(code)
        tree = javalang.parse.parse(original_code)
        flatten = []
        for path, node in tree:
            flatten.append({'path': path, 'node': node})
        wf.writelines(str(flatten))
        # wf.write(str(flatten))
    # with open(path_w) as f:
    #     f.write(path_w)
        # print(flatten)

        # wf.write(json.dumps(flatten))
            # print(path, node)
            # print('------------')
            # print('')
            # wf.write(json.dumps(node))
            # wf.write('\n')
            # print(path)
            # print(node)
        # print(tree)
        # print(tree.type[0])
        # for line in tqdm(original_code):
            # print(line)
            # print('before(line):{}'.format(line))
            # code = line.strip()
            # code = line
            # print('code:{}'.format(code))
            # print('after(line):{}'.format(line))
            # tokens = javalang.tokenizer.tokenize(code)
            # print('tokens:{}'.format(tokens))
            # token_list = list(javalang.tokenizer.tokenize(code))
            # print('token_list:{}'.format(token_list))
            # length = len(token_list)
            # print('length:{}'.format(length))
            # parser = javalang.parser.Parser(tokens)
            # print('parser:{}'.format(parser))



if __name__ == '__main__':
    # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_
    print(sys.argv)
    process_source(sys.argv[1], 'source.code')
    # generate ast file for source code
    get_ast('source.code', sys.argv[2])
    # get_ast('train.source', sys.argv[2])