import basicTypes
import struct

class Literal:
    def __init__(self, x, h = False):
        self.value = x
        self.type = basicTypes.word
        self.isHex = h

    def __or__(self,other):
        return Literal(self.value | other.value, self.isHex or other.isHex)

    def __add__(self,other):
        return Literal(self.value + other.value, self.isHex or other.isHex)

    def __lshift__(self,other):
        return Literal(self.value << other.value, self.isHex)

    def __lt__(self,other):
        return self.value < other

    def __neg__(self):
        return Literal(-self.value, self.isHex)

    def __eq__(self, other):
        try:
            return self.value == other.value
        except:
            return self.value == other


    def __format__(self, spec):
        if self.type == basicTypes.address and self.value == 0:
            return 'None'

        topbyte = self.value >> 24
        if topbyte | 0x80 in range(0xbd,0xc8):
            return '{:.4f}'.format(struct.unpack('>f',self.value.to_bytes(4,byteorder='big'))[0])

        global noooooo

        if 'h' in spec or topbyte == 0x80 or self.type == basicTypes.address:
            return hex(self.value)

        # silly heuristic
        h = hex(self.value)
        d = str(self.value)
        return h if (h.count('0')-1)/len(h) > d.count('0')/len(d) else d

    def __repr__(self):
        return 'Literal({:#x})'.format(self.value)

class Symbolic:
    pass

class Symbol(Symbolic):
    def __init__(self, sym, d = basicTypes.unknown):
        self.name = sym
        self.type = d
        
    def negated(self):
        return Symbol('not {}'.format(self))    # not a good solution

    def toHex(self):
        return self

    def __eq__(self,other):
        return isinstance(other,Symbol) and self.name == other.name

    def __format__(self, spec):
        return self.name

    def __repr__(self):
        return 'Symbol({},{})'.format(self.name,self.type)
    
class Expression(Symbolic):

    logicOpposite = {'==':'!=', '!=':'==', '>':'<=', '<':'>=', '<=':'>', '>=':'<'}

    def __init__(self, op, args, fmt = basicTypes.unknown, constant = None):
        self.op = op
        self.args = args
        self.type = fmt
        self.constant = constant

    def __format__(self, spec):
        if self.op == '@':
            return '{}({})'.format(self.args[0], self.args[1])
        if '!' in spec and self.type == basicTypes.boolean:
            sep = ' {} '.format(Expression.logicOpposite[self.op])
        elif self.op in '* / **'.split():
            sep = self.op
        else:
            sep = ' {} '.format(self.op)

        if 'h' in spec or self.op in '|&^' or self.type == basicTypes.address:
            inner = '{:ph}'
        else:
            inner = '{:p}'
            
        try:
            return ('({}{})' if 'p' in spec else '{}{}').format(sep.join(inner.format(a) for a in self.args),
                                                '{{}}{}'.format(inner).format(sep,self.constant) if self.constant else '')
        except:
            print('error formatting', repr(self))
            raise

    def negated(self):
        if self.type == basicTypes.boolean:
            return Expression(Expression.logicOpposite[self.op], self.args, self.type)
        raise Exception("Can't negate non-logical expression")

    def __repr__(self):
        return "Expression({}, {}{})".format(self.op, ', '.join(repr(a) for a in self.args),
                                                    '; {}'.format(self.constant) if self.constant else '')

    opLambdas = {
        '+' : lambda x, y: x + y,
        '*' : lambda x, y: x * y,
        '-' : lambda x, y: x - y,
        '/' : lambda x, y: x / y,
        '>>': lambda x, y: x >> y,
        '<<': lambda x, y: x << y,
        '|' : lambda x, y: x | y,
        '^' : lambda x, y: x ^ y,
        '&' : lambda x, y: x & y,
    }

    opIdentities = {
        '+': 0,
        '*': 1,
        '|': 0
    }

    @staticmethod
    def build(op, left, right, flop = False):

        if op == 'NOR':     #why is this a thing
            return Expression('~',[Expression.build('|', left, right, flop)])

        if isinstance(left, Literal):
            if isinstance(right, Literal):
                #two literals, completely reducible
                return Literal(Expression.opLambdas[op](left.value, right.value))
            left, right = right, left
        #left is not a literal, right may be

        if op == '*' and left == right:
            return Expression('**', [left], constant=Literal(2), fmt=basicTypes.single if flop else basicTypes.word)

        if op in ['==', '!='] and isinstance(right, Literal):
            if left.type == basicTypes.boolean and right == 0:
                return left if op == '!=' else left.negated()
            if basicTypes.isAddressable(left.type) and right == 0:
                return left
            if isinstance(left.type, basicTypes.EnumInstance):
                right.type = left.type

        if op == '<<' and right.value < 8:
            op, right = '*', Literal(2**right.value)

        if op in '+*|':
            return Expression.arithmeticMerge(op, [left, right], flop)
        else:
            new = Expression(op, [left, right])

        if op in '< > <= >= == !='.split():
            new.type = basicTypes.boolean
            
        return new

    @staticmethod
    def arithmeticMerge(op, args, flop = False):
        symbols = []
        newConstant = Expression.opIdentities[op]
        for a in args:
            if isinstance(a, Expression) and a.op == op:
                symbols.extend(a.args)
                if a.constant:
                    newConstant = Expression.opLambdas[op](newConstant, a.constant.value)
            elif isinstance(a, Literal):
                newConstant = Expression.opLambdas[op](newConstant, a.value)
            else:
                symbols.append(a)
        newConstant = None if newConstant == Expression.opIdentities[op] else Literal(newConstant)
        if symbols:     #in case I add symbolic cancellation later
            if len(symbols) == 1 and not newConstant:
                return symbols[0]
            else:
                #multiple expressions summed
                if flop:
                    newType = basicTypes.single
                else:
                    newType = basicTypes.word
                    for s in symbols:
                        if basicTypes.isAddressable(s.type):
                            newType = basicTypes.address
                            break
                return Expression(op, symbols, constant=newConstant, fmt=newType)
        elif newConstant:
            return newConstant
        else:
            return Literal(Expression.opIdentities[op])
