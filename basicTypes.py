from collections import namedtuple

Pointer = namedtuple('Pointer', 'pointedType target')
Array = namedtuple('Array', 'pointedType length')
Stack = namedtuple('Stack', 'offset')
StructType = namedtuple('StructType', 'name size members')
EnumType = namedtuple('EnumType', 'name base values')
Flag = namedtuple('Flag', 'base bits')
FunctionSignature = namedtuple('FunctionSignature', 'name args')
Variable = namedtuple('Variable', 'name type')

class Primitive:
    lookup = {}

    def __init__(self, name, size):
        self.size = size
        self.name = name
        Primitive.lookup[name] = self

    def __repr__(self):
        return self.name

unknown = Primitive('unknown', 0)
bad = Primitive('bad', 0)
boolean = Primitive('boolean', 4)  # 1/0 boolean, full word
word = Primitive('word', 4)
address = Primitive('address', 4)
short = Primitive('short', 2)
ushort = Primitive('ushort', 2)
byte = Primitive('byte', 1)
ubyte = Primitive('ubyte', 1)

single = Primitive('single', 4)
double = Primitive('double', 8)

def getCode(t):
    codeTable = {unknown: '', bad: 'x', boolean: '?', word: 'w', address: 'p',
                 short: 'h', ushort: 'H', byte: 'b', ubyte: 'B'}
    if isinstance(t, Primitive):
        return codeTable[t]
    return 'p'