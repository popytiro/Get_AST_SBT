import javalang
import json
from tqdm import tqdm    # tqdmを使用して, プログレスバーを導入．ange()関数の値をtqdm()関数に与えるだけでプログレスバーを表示出来る．
import collections
import sys


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


# astの取得
# souce.codeからast.jsonを生成
def get_ast(file_name, w):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(w, 'w+', encoding='utf-8') as wf:
        ign_cnt = 0
        for line in tqdm(lines):
            # print('before(line):{}'.format(line))
            code = line.strip()
            # print('after(code):{}'.format(code))
            tokens = javalang.tokenizer.tokenize(code)
            # print('tokens:{}'.format(tokens))
            token_list = list(javalang.tokenizer.tokenize(code))
            # print('tokens_list:{}'.format(token_list))
            length = len(token_list)
            # print('length:{}'.format(length))
            parser = javalang.parser.Parser(tokens)
            # print('parser:{}'.format(parser))
            try:
                # treeに入るもの(NoneType,AnnotationDeclaration,MethodDeclaration,FieldDeclaration,ConsttuctorDeclaration)
                tree = parser.parse_member_declaration()
            # print('hello')
                # or parser.parse_method_or_field_declaraction()
                # tree = parser.parse_method_declarator_rest()
                # print(tree)
            # except javalang.parser.JavaSyntaxError as e:
            #     print(e)
            except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                print(code)
                continue
            # except:
            #     print('error発生')
            # else:
            #     print('正常終了')


            flatten = []
            for path, node in tree:
                # print('path:{}'.format(path))
                # print('node:{}'.format(node))
                flatten.append({'path': path, 'node': node})

            ign = False
            outputs = []
            stop = False
            for i, Node in enumerate(flatten):
                # print('i:{}'.format(i))
                d = collections.OrderedDict()
                path = Node['path']
                node = Node['node']
                children = []
                for child in node.children:
                    child_path = None
                    if isinstance(child, javalang.ast.Node):
                        child_path = path + tuple((node,))
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                                children.append(j)
                    if isinstance(child, list) and child:
                        child_path = path + (node, child)
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path']:
                                children.append(j)
                d["id"] = i
                d["type"] = str(node)
                if children:
                    d["children"] = children
                value = None
                if hasattr(node, 'name'):
                    value = node.name
                elif hasattr(node, 'value'):
                    value = node.value
                elif hasattr(node, 'position') and node.position:
                    for i, token in enumerate(token_list):
                        if node.position == token.position:
                            pos = i + 1
                            value = str(token.value)
                            while (pos < length and token_list[pos].value == '.'):
                                value = value + '.' + token_list[pos + 1].value
                                pos += 2
                            break
                elif type(node) is javalang.tree.This \
                        or type(node) is javalang.tree.ExplicitConstructorInvocation:
                    value = 'this'
                elif type(node) is javalang.tree.BreakStatement:
                    value = 'break'
                elif type(node) is javalang.tree.ContinueStatement:
                    value = 'continue'
                elif type(node) is javalang.tree.TypeArgument:
                    value = str(node.pattern_type)
                elif type(node) is javalang.tree.SuperMethodInvocation \
                        or type(node) is javalang.tree.SuperMemberReference:
                    value = 'super.' + str(node.member)
                elif type(node) is javalang.tree.Statement \
                        or type(node) is javalang.tree.BlockStatement \
                        or type(node) is javalang.tree.ForControl \
                        or type(node) is javalang.tree.ArrayInitializer \
                        or type(node) is javalang.tree.SwitchStatementCase:
                    value = 'None'
                elif type(node) is javalang.tree.VoidClassReference:
                    value = 'void.class'
                elif type(node) is javalang.tree.SuperConstructorInvocation:
                    value = 'super'
                # ここからfor method declaration
                # elif type(node) is javalang.tree.TypeDeclaration:
                #     value = 'TypeDeclaration'
                # elif type(node) is javalang.tree.MethodDeclaration:
                #     value = 'MethodDeclaration'
                # ここまで


                if value is not None and type(value) is type('str'):
                    d['value'] = value
                if not children and not value:
                    # print('Leaf has no value!')
                    print(type(node))
                    print('aaa')
                    print(code)
                    print('aaa')
                    ign = True
                    ign_cnt += 1
                    # break
                outputs.append(d)
            if not ign:
                wf.write(json.dumps(outputs))
                wf.write('\n')
    print(ign_cnt)


if __name__ == '__main__':
    # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_
    print(sys.argv)
    process_source(sys.argv[1], 'source.code')
    # generate ast file for source code
    get_ast('source.code', sys.argv[2])
