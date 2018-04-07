import sys

tokens = sys.stdin.read()

last = ''
indent = -1
stack = []

def peek(tok):
    return stack[-1] == tok

def down_indent(tok):
    global indent, stack, last
    stack = stack[:-1]
    indent -= 1
    print tok+'\n' + ' '*4*indent,

def up_indent(tok):
    global indent, stack, last
    stack.append(tok)
    indent += 1
    print tok+'\n' + ' '*4*indent,

for i in tokens:
    if i in ['[']:
        up_indent(i)
    elif i == ']' and peek('['):
        down_indent(i)
    elif i == ')' and peek('('):
        down_indent(i)
    elif i == ',' and last in [']', ')', '(', '[']:
        print i+'\n' + ' '*(4*indent-1),
    else:
        sys.stdout.write(i)
    last = i

print ''
