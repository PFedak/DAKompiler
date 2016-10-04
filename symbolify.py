import struct
import itertools
from collections import defaultdict
from functools import reduce

from instruction import *

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
        topbyte = self.value >> 24
        if topbyte | 0x80 in [0xbf, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7]:
            return str(struct.unpack('>f',self.value.to_bytes(4,byteorder='big'))[0])

        if 'h' in spec or topbyte == 0x80:
            return hex(self.value)
        else:
            return str(self.value)

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

    def __init__(self, op, args, fmt = basicTypes.unknown):
        self.op = op
        self.args = args
        self.type = fmt

    def __format__(self, spec):
        if self.op == '@':
            return '{}({})'.format(self.args[0], self.args[1])
        if '!' in spec and self.type == basicTypes.boolean:
            sep = ' {} '.format(Expression.logicOpposite[self.op])
        elif self.op in '* / **'.split():
            sep = self.op
        else:
            sep = ' {} '.format(self.op)

        if 'h' in spec or self.op in '|&^':
            inner = '{:ph}'
        else:
            inner = '{:p}'

        try:
            return ('({})' if 'p' in spec else '{}').format(sep.join(inner.format(a) for a in self.args))
        except:
            print('error formatting', repr(self))
            raise

    def negated(self):
        if self.type == basicTypes.boolean:
            return Expression(Expression.logicOpposite[self.op], self.args, self.type)
        raise Exception("Can't negate non-logical expression")

    def __repr__(self):
        return "Expression({}, {})".format(self.op, ', '.join(repr(a) for a in self.args))

    opLambdas = {
        '+' : lambda x, y: x + y,
        '*' : lambda x, y: x * y,
        '-' : lambda x, y: x - y,
        '/' : lambda x, y: x / y,
        '>>': lambda x, y: x >> y,
        '<<': lambda x, y: x << y,
        '|' : lambda x, y: x | y,
        '^' : lambda x, y: x ^ y,
        '&' : lambda x, y: x & y
    }

    opIdentities = {
        '+': 0,
        '*': 1,
        '|': 0
    }

    @staticmethod
    def build(op, left, right, flop = False):

        if isinstance(left, Literal):
            if isinstance(right, Literal):
                #two literals, completely reducible
                return Literal(Expression.opLambdas[op](left.value, right.value))
            left, right = right, left
        #left is not a literal, right may be

        if op == '*' and left == right:
            return Expression('**', [left, Literal(2)], basicTypes.single if flop else basicTypes.word)

        if op in '+*|':
            return Expression.arithmeticMerge(op, left, right, flop)
        else:
            new = Expression(op, [left, right])

        if op in '< > <= >= == !='.split():
            new.type = basicTypes.boolean
            if left.type == basicTypes.boolean and right == Literal(0):
                if op == '!=':
                    return left
                if op == '==':
                    return Expression(Expression.logicOpposite[left.op], left.args)
        return new

    @staticmethod
    def arithmeticMerge(op, left, right, flop):
        toMerge = []
        if isinstance(left, Expression) and left.op == op:
            toMerge.extend(left.args)
        else:
            toMerge.append(left)
        if isinstance(right, Expression) and right.op == op:
            toMerge.extend(right.args)
        else:
            toMerge.append(right)
        #separate literal values to combine
        byType = {True:[], False:[]}
        for x in toMerge:
            byType[isinstance(x,Literal)].append(x)
        if byType[True]:
            combined = Literal(reduce(Expression.opLambdas[op], [l.value for l in byType[True]]))
            if combined.value != Expression.opIdentities[op]:
                byType[False].append(combined)
        if byType[False]:
            if len(byType[False]) == 1:
                return byType[False][0]
            else:
                return Expression(op, byType[False], basicTypes.single if flop else basicTypes.word)
        else:       #all literals, sum was zero, seems unlikely
            return Literal(0)

def extend(const):
    if const < 0x8000:
        return const
    else:
        return const-0x10000

def assignReg(w, fmt ,op):
    def foo(instr, history):
        value = get_value(history, instr)
        toWrite = get_toWrite(instr)
        return (InstrResult.register, history.write(toWrite, value))

    def get_toWrite(instr):
        if w == 'T':
            return instr.targetReg
        elif w == 'D':
            return instr.destReg
        elif w == 'F':
            return instr.fd
        elif w == 'C':
            return SpecialRegister.Compare
        raise Error("Bad w")

    def get_value(history, instr):
        if fmt == 'S+xI':
            return Expression.build(op, history.read(instr.sourceReg, basicTypes.word), Literal(instr.immediate))
        elif fmt == 'S+I':
            return Expression.build(op, history.read(instr.sourceReg, basicTypes.word), Literal(extend(instr.immediate)))
        elif fmt == 'f(I)':
            return Literal(op(instr.immediate))
        elif fmt == 'F+F':
            return Expression.build(op, history.read(instr.fs, basicTypes.single), 
                                        history.read(instr.ft, basicTypes.single), flop=True)
        elif fmt == '@F':
            if op == 'id':
                return history.read(instr.fs)
            else:
                return Expression.build('@', op, history.read(instr.fs))
        elif fmt == 'T<<A':
            return Expression.build(op, history.read(instr.targetReg), Literal(instr.shift))
        elif fmt == 'S+T':
            return Expression.build(op, history.read(instr.sourceReg, basicTypes.word),
                                        history.read(instr.targetReg, basicTypes.word))
        raise Error("Bad format")

    return foo

def loadMemory(datatype):
    #   TODO:
    #       account for stack reads at different offsets (ie, loading the low short of a word)
    #       add datatype memory
    def foo(instr, history):
        if instr.sourceReg == Register.SP:
            #TODO account for more general reads (ie, just the lower bytes of a word)
            value = history.read(basicTypes.Stack(extend(instr.immediate)), datatype)  
        else:
            address = Expression.build('+', history.read(instr.sourceReg, basicTypes.word), 
                                            Literal(extend(instr.immediate)))
            value = history.lookupAddress(datatype, address)
        return InstrResult.register, history.write(instr.targetReg, value)
    return foo

def storeMemory(datatype):
    def foo(instr, history):
        value = history.read(instr.targetReg, datatype)
        if instr.sourceReg == Register.SP:
            return InstrResult.register, history.write(basicTypes.Stack(extend(instr.immediate)), value) 
        else:
            return InstrResult.write, value, history.lookupAddress(datatype, 
                Expression.build('+', history.read(instr.sourceReg, basicTypes.word), 
                                      Literal(extend(instr.immediate))))
    return foo

def branchMaker(comp, withZero, likely = False):
    def doBranch(instr, history):
        if instr.opcode == MainOp.BEQ and instr.sourceReg == instr.targetReg:
            return InstrResult.jump, None, extend(instr.immediate)
        return (InstrResult.likely if likely else InstrResult.branch, 
                    Expression.build(comp, 
                        history.read(instr.sourceReg, basicTypes.word), 
                        history.read(Register.R0 if withZero else instr.targetReg, basicTypes.word)), 
                    extend(instr.immediate))
    return doBranch

def MFC_python(instr, history):
    if instr.cop == 0:
        raise Exception("COP0 unimplemented")
    if instr.cop == 1:
        return InstrResult.register, history.write(instr.targetReg, history.read(instr.fs))
    
def MTC_python(instr, history):
    if instr.cop == 0:
        raise Exception("COP0 unimplemented")
    if instr.cop == 1:
        return InstrResult.register, history.write(instr.fs, history.read(instr.targetReg))

InstrResult = Enum('InstrResult', 'none register read write function branch likely jump end unhandled')

conversionList = {
    MainOp.JAL: lambda instr,regs: (InstrResult.function, 0x80000000 + instr.target),
    #note that branches are written negated, the 'if' code is the code immediately following
    MainOp.BEQ: branchMaker('!=', withZero = False),
    MainOp.BEQL: branchMaker('!=', withZero = False, likely = True),
    MainOp.BNE: branchMaker('==', False),
    MainOp.BNEL: branchMaker('==', False, True),
    MainOp.BLEZ: branchMaker('>', True),
    MainOp.BGTZ: branchMaker('<=', True),

    MainOp.ADDIU: assignReg('T','S+I','+'),
    MainOp.SLTI: assignReg('T','S+I','<'),
    MainOp.SLTIU: assignReg('T','S+xI','<'),
    MainOp.ANDI: assignReg('T','S+xI','&'),
    MainOp.ORI: assignReg('T','S+xI','|'),
    MainOp.XORI: assignReg('T','S+xI','^'),
    MainOp.LUI: assignReg('T','f(I)',lambda x: x << 16),

    MainOp.LB: loadMemory(basicTypes.byte),
    MainOp.LH: loadMemory(basicTypes.short),
    MainOp.LW: loadMemory(basicTypes.word),
    MainOp.LBU: loadMemory(basicTypes.ubyte),
    MainOp.LHU: loadMemory(basicTypes.ushort),
    MainOp.SB: storeMemory(basicTypes.byte),
    MainOp.SH: storeMemory(basicTypes.short),
    MainOp.SW: storeMemory(basicTypes.word),
    MainOp.LWC1: loadMemory(basicTypes.single),
    MainOp.SWC1: storeMemory(basicTypes.single),
    MainOp.SDC1: storeMemory(basicTypes.double),
    MainOp.LDC1: loadMemory(basicTypes.double),
    
    RegOp.SLL: assignReg('D','T<<A','<<'),
    RegOp.SRL: assignReg('D','T<<A','>>'),
    RegOp.SRA: assignReg('D','T<<A','>a'),
    RegOp.JR: lambda instr, history: (InstrResult.end,) if instr.sourceReg == Register.RA else (InstrResult.unhandled, Expression.build('@','JR',history.read(instr.sourceReg))),
    RegOp.JALR: lambda instr, history: (InstrResult.unhandled, 'JALR', 
                                            history.read(instr.sourceReg, basicTypes.address)), 
    RegOp.ADD: assignReg('D','S+T','+'),
    RegOp.ADDU: assignReg('D','S+T','+'),
    RegOp.SUB: assignReg('D','S+T','-'),
    RegOp.SUBU: assignReg('D','S+T','-'),
    RegOp.DIV: assignReg('D','S+T','/'),
    RegOp.AND: assignReg('D','S+T','&'),
    RegOp.OR: assignReg('D','S+T','|'),
    RegOp.XOR: assignReg('D','S+T','^'),
    RegOp.SLT: assignReg('D','S+T','<'),
    RegOp.SLTU: assignReg('D','S+T','<'),

    FloatOp.ADD: assignReg('F','F+F','+'),
    FloatOp.SUB: assignReg('F','F+F','-'),
    FloatOp.MUL: assignReg('F','F+F','*'),
    FloatOp.DIV: assignReg('F','F+F','/'),
    FloatOp.SQRT: assignReg('F','@F','sqrt'),
    FloatOp.ABS: assignReg('F','@F','abs'),
    FloatOp.MOV: assignReg('F','@F','id'), #lol
    FloatOp.NEG: assignReg('F','@F','neg'),
    FloatOp.ROUND_W: assignReg('F','@F','round'),
    FloatOp.TRUNC_W: assignReg('F','@F','trunc'),
    FloatOp.CEIL_W: assignReg('F','@F','ceil'),
    FloatOp.FLOOR_W: assignReg('F','@F','floor'),
    FloatOp.CVT_S: assignReg('F','@F','id'),
    FloatOp.CVT_D: assignReg('F','@F','id'),
    FloatOp.CVT_W: assignReg('F','@F','id'),
    FloatOp.C_EQ: assignReg('C', 'F+F', '=='),
    FloatOp.C_LE: assignReg('C', 'F+F', '<='),
    FloatOp.C_LT: assignReg('C', 'F+F', '<'),
    CopOp.MFC: MFC_python,
    CopOp.MTC: MTC_python,
    CopOp.BCF: lambda instr,history: (InstrResult.branch, history.read(SpecialRegister.Compare), extend(instr.target)),
    CopOp.BCT: lambda instr,history: (InstrResult.branch, history.read(SpecialRegister.Compare).negated(), extend(instr.target)),
    CopOp.BCFL: lambda instr,history: (InstrResult.likely, history.read(SpecialRegister.Compare), extend(instr.target)),
    CopOp.BCTL: lambda instr,history: (InstrResult.likely, history.read(SpecialRegister.Compare).negated(), extend(instr.target)),
    CopOp.CFC: lambda instr,regs: None,
    CopOp.CTC: lambda instr,regs: None,

    SpecialOp.NOP: lambda instr,history: (InstrResult.none,),
    SpecialOp.BGEZL: branchMaker('<', withZero = True, likely = True),
    SpecialOp.BGEZ: branchMaker('<', withZero = True),
    SpecialOp.BLTZ: branchMaker('>=', withZero = True),
}

class Branch(dict):
    """A particular path taken through the program"""
    def __init__(self, choices = {}, lineNumber = 0):
        self.line = lineNumber
        super(Branch, self).__init__(choices)
        self.hashValue = hash(','.join('%x%s' % (c,'T' if self[c] else 'F') for c in sorted(self.keys())))

    def branchOff(self, split, stayed, currLine = 0):
        copy = self.copy()
        copy[split] = stayed
        return Branch(copy, currLine)

    def implies(self, other):
        for ch in other:
            if ch in self and self[ch] == other[ch]:
                continue
            else:
                return False
        return True

    def isCompatibleWith(self, other):
        for ch in self:
            if ch in other and self[ch] != other[ch]:
                return False
        return True

    def tryMerge(self, other):
        diff = -1
        for ch in self:
            if not ch in other:
                return -1
            if other[ch] != self[ch]:
                if diff >= 0:
                    return -1
                else:
                    diff = ch
        return diff

    def without(self, avoid):
        return Branch({x:self[x] for x in self if x not in avoid}, self.line)

    def __hash__(self):
        return self.hashValue

    def __eq__(self, other):
        return self.implies(other) and other.implies(self)

    def __repr__(self):
        return '(%s)' % ' and '.join([('not 'if not self[c] else '') + 'cmp_%s' % hex(4*c)[2:] for c in self])



class Context:
    def __init__(self, branchList = None, line = 0):
        """Merge the list of branches into something a bit more friendly."""
        self.line = line
        if not branchList:
            self.cnf = [Branch()]
            return
        allChoices = set()
        for br in branchList:
            for ch in br:
                allChoices.add(ch)
        allChoices = sorted(allChoices)
        #kinda sorta Quine-McCluskey
        byLenbyOnes = defaultdict(lambda: defaultdict(dict))    
        prime = set()
        for br in branchList:
            byLenbyOnes[len(br)][sum(1 for ch in br if br[ch])][br]= False
        bottom = max(byLenbyOnes.keys())
        for L in range(bottom, -1, -1):
            for ones in range(L+1):
                for lower in byLenbyOnes[L][ones]:
                    for upper in byLenbyOnes[L][ones+1]:
                        # find True/False differences 
                        index = lower.tryMerge(upper)
                        if index >= 0:
                            byLenbyOnes[L][ones][lower] = True
                            byLenbyOnes[L][ones+1][upper] = True
                            newOnes = ones-1 if lower[index] else ones
                            byLenbyOnes[L-1][newOnes][lower.without([index])] = False
                    if not byLenbyOnes[L][ones][lower]:
                        found = False
                        for upper in itertools.chain(byLenbyOnes[L-1][ones-1], byLenbyOnes[L-1][ones]):
                            # check for previous merges this falls under
                            if upper.implies(lower):
                                found = True
                                break
                        if not found:
                            prime.add(lower)
                    
        #TODO actually finish the algorithm
        self.cnf = list(prime)

    def implies(self, other):
        """Check if this context implies the other one, and if so return the relative conditions"""

        # There are combinations of context that will make the "relative" result nonsensical, 
        #   I'm not certain if they will actually appear. For instance, 
        #   self = (x and not y and z) or (x and y and not z)
        #   other = (x and not y) or (x and y)
        #   self does imply other, but relative will be an empty context

        relative = []
        for base in self.cnf:
            options = [target for target in other.cnf if base.implies(target)]
            if options:
                relative.append(min([base.without(target) for target in options], key=len))
            else:
                return False, None
        return True, Context(relative)

    def isCompatibleWith(self, other):
        """Check if any branch could satisfy both contexts"""
        for base in self.cnf:
            if [t for t in other.cnf if base.isCompatibleWith(t)]:
                return True
        return False

    def isTrivial(self):
        return len(self.cnf) == 1 and not self.cnf[0]

    def __repr__(self):
        return ' or '.join(str(br) for br in self.cnf)

class VariableState:
    def __init__(self, name, value, context):
        self.name = name
        self.value = value
        self.context = context
        self.explicit = False

    def __repr__(self):
        return '{} = {} ({})'.format(self.name, self.value, self.context)

class VariableHistory:
    def __init__(self, bindings, args = []):
        self.states = defaultdict(list)
        self.bindings = bindings
        self.argList = []           #arguments beyond the given ones
        self.now = Context([Branch()])
        self.write(Register.R0, Literal(0))
        self.write(Register.SP, Symbol('SP'))
        self.write(Register.RA, Symbol('RA'))
        self.write('CC', Symbol('bad_CC'))
        for reg, name, fmt in args:
            showName = name if name else VariableHistory.getName(arg)
            self.argList.append(reg)
            self.write(reg, Symbol(showName, fmt))

    def read(self, var, fmt = basicTypes.unknown):
        if var in self.states:
            uncertain = False
            for st in reversed(self.states[var]):
                if self.now.implies(st.context)[0]: # this state definitely occurred
                    if uncertain:
                        st.explicit = True
                        break
                    else:
                        if st.value.type == basicTypes.unknown:
                            st.value.type = fmt
                        return st.value
                elif self.now.isCompatibleWith(st.context):
                    st.explicit = True
                    uncertain = True
            return Symbol(VariableHistory.getName(var), fmt)
        else:
            symName = VariableHistory.getName(var)
            if VariableHistory.couldBeArg(var): 
                self.argList.append(var)
                symName = 'arg_' + symName
            self.states[var].append(VariableState(self.getName(var), Symbol(symName, fmt), self.now))
            return self.states[var][-1].value
        

    def write(self, var, value):
        self.states[var].append(VariableState(self.getName(var), value, self.now))
        return self.states[var][-1]

    def markBad(self, var):
        self.write(var, Symbol('bad_%s' % VariableHistory.getName(var), basicTypes.bad))    

    def isValid(self, var):
        """Determine if reading from the variable makes sense, mainly for function arguments"""

        # has the function been touched at all?
        if var not in self.states:  
            return False
        # have we marked it as "bad" - 
        try:
            return self.states[var][-1].value.type != basicTypes.bad
        except:
            # TODO: check that a value is set along all branches, unsure how often this will come up
            return True

    def lookupAddress(self, fmt, address):
        if isinstance(address, Literal):
            if address.value in self.bindings['globals']:
                base = Symbol(*self.bindings['globals'][address.value])
                if self.isAddressable(base.type):
                    return self.subLookup(fmt, base.name, base.type, 0)
                else:
                    return base
            bestOffset = max(x for x in self.bindings['globals'] if x <= address.value)
            base = Symbol(*self.bindings['globals'][bestOffset])
            relOffset = address.value - bestOffset
            if relOffset >= self.getSize(base.type):
                return Symbol('{}({:#010x})'.format(fmt, address.value), fmt)
            else:
                return self.subLookup(fmt, base.name, base.type, relOffset)

        base = None
        memOffset = 0
        others = []            
        if isinstance(address, Symbol):
            if isinstance(address.type, basicTypes.Pointer):
                return Symbol(address.type.target if address.type.target else address.name, address.type.pointedType)
            if self.isAddressable(address.type):
                base = address
        elif address.op == '+':
            for term in address.args:
                if self.isAddressable(term.type):
                    if base:
                        raise Exception('adding structs')
                    base = term
                elif isinstance(term, Literal):
                    memOffset = term.value
                else:
                    others.append(term)

        if not base:
            #check for trig lookup
            if fmt == basicTypes.single and memOffset in self.bindings['trigtables']:
                try:
                    angle = others[0].args[0].args[0]
                    return Symbol('{}Table({})'.format(self.bindings['trigtables'][memOffset], angle))
                except:
                    pass

            # no idea what we are looking at, process it anyway
            return Symbol('{}({:h})'.format(fmt, address), fmt)
        if memOffset >= self.getSize(base.type):
            raise Exception('trying to look up address %#x in %s (only %#x bytes)' % (memOffset, base.type, self.getSize(base.type)))

        # determine which member struct we are trying to access
        return self.subLookup(fmt, base.name, base.type, memOffset, others)

        
    def subLookup(self, fmt, prefix, superType, address, others = []):
        """Recursively find data at the given address from the start of a type"""
        if isinstance(superType, basicTypes.Array):
            if others:
                #TODO check the variable offset for multiplication/shift, and undo it
                index = Expression('+', others + [Literal(address)])
                return Symbol('{}({} + {:h})'.format(fmt, prefix, index), superType.pointedType)
            else:
                index = address//self.getSize(superType.pointedType)
                return Symbol('{}[{:d}]'.format(prefix, index), superType.pointedType)
        if isinstance(superType, basicTypes.Pointer):
            return Symbol(superType.target if superType.target else prefix, superType.pointedType)
        if isinstance(superType, str) and superType in self.bindings['structs']:
            members = self.bindings['structs'][superType].members
            bestOffset = max(x for x in members if x <= address)
            base = Symbol(*members[bestOffset])
            if address < bestOffset + self.getSize(base.type):
                if self.isAddressable(base.type):
                    return self.subLookup(fmt, '{}.{}'.format(prefix, base), base.type, address - bestOffset, others)
                if not others:
                    #TODO account for reading the lower short of a word, etc.
                    return Symbol('{}.{}'.format(prefix, base), base.type)
        # nothing matched
        if others:
            return Symbol('{}({} + {:h})'.format(fmt, prefix, Expression.build('+', Expression('+',others), Literal(address))), fmt)
        else:
            return Symbol('{}.{}_{:#x}'.format(prefix, basicTypes.getCode(fmt), address), fmt)

    @staticmethod
    def getName(var):
        try:
            return var.name 
        except:
            try:
                return 'stack_%x' % var.offset
            except:
                return var

    def getSize(self, t):
        try:
            return t.size
        except:
            if isinstance(t, basicTypes.Pointer):
                return 4
            if isinstance(t, basicTypes.Flag):
                return t.base.size
            if isinstance(t, basicTypes.Array):
                return t.length * self.getSize(t.pointedType) 
            if t in self.bindings['structs']:
                return self.bindings['structs'][t].size
            if t in self.bindings['enums']:
                return self.getSize(self.bindings['enums'][t].base)
        print('failed to size', t)

    def isAddressable(self, t):
        if isinstance(t, basicTypes.Primitive):
            return False
        if isinstance(t, basicTypes.Pointer) or isinstance(t, basicTypes.Array):
            return True
        if isinstance(t, str):
            # assume anything unspecified is addressible
            return t not in self.bindings['enums']
        
        #flags, anything else
        return False

    @staticmethod
    def couldBeArg(var):
        if var in [Register.A0, Register.A1, Register.A2, Register.A3, FloatRegister.F12, FloatRegister.F14]:
            return True
        try:
            if var.offset in range(0x10,0x20):
                return True
        except:
            pass
        return False

class CodeBlock:
    def __init__(self, context, parent = None, relative = None):
        self.code = []
        self.context = context
        self.parent = parent
        self.relative = relative
        self.children = []

def makeSymbolic(name, mipsData, bindings, arguments = []):
    """Produce symbolic representation of the logic of a MIPS function"""

    address, mips, loops = mipsData

    baseBranch = Branch()
    currContext = Context([baseBranch])   #no branches yet
    branchList = [baseBranch]       #branches and their current lines
    updates = set()
    booleans = {}                   #will hold the symbols associated with branches
    delayed = None


    mainCode = CodeBlock(currContext)
    currBlock = mainCode

    history = VariableHistory(bindings, arguments)

    for lineNum, instr in enumerate(mips):
        if lineNum in updates:
            newContext = Context([b for b in branchList if 0 <= b.line <= lineNum], lineNum)
            newParent = currBlock
            while True:
                imp, rel = newContext.implies(newParent.context)
                if imp:
                    break
                else:
                    newParent = newParent.parent
            currBlock = CodeBlock(newContext, newParent, rel)
            newParent.children.append(currBlock)
            history.now = newContext
            #TODO prune now-irrelevant choices from branches so this doesn't take forever on long functions
        try:
            result = conversionList[instr.opcode](instr, history)
        except ValueError:
            currBlock.code.append((InstrResult.unhandled, instr))
        else:
            if result[0] in [InstrResult.branch, InstrResult.likely, InstrResult.jump]:
                if result[1]:
                    booleans[lineNum] = result[1]
                delayed = (result[0], lineNum + 1 + result[-1])
                continue
            elif result[0] in [InstrResult.function, InstrResult.end]:
                delayed = result
                continue
            elif result[0] != InstrResult.none:
                currBlock.code.append(result)
        
        if delayed:
            if delayed[0] in [InstrResult.branch, InstrResult.likely, InstrResult.jump]:
                branchType, branchDest = delayed
                currBranches = [x for x in branchList if 0 <= x.line <= lineNum-1]
                if branchType == InstrResult.jump:
                    for b in currBranches:
                        b.line = branchDest
                        updates.add(branchDest)
                else:
                    for b in currBranches:
                        b.line = -1
                        branchList.append(b.branchOff(lineNum-1, True, lineNum+1))
                        branchList.append(b.branchOff(lineNum-1, False, branchDest))
                        updates.add(lineNum+1)
                        updates.add(branchDest)
            elif delayed[0] == InstrResult.function:
                argList = []
                funcCall = delayed[1]
                if funcCall in bindings['functions']:
                    title = bindings['functions'][funcCall].name
                    for reg, argName, fmt in bindings['functions'][funcCall].args:
                        argList.append((argName, history.read(reg, fmt)))
                        history.markBad(reg)
                else:
                    title = 'fn%06x' % funcCall
                    for reg in [Register.A0, FloatRegister.F12, Register.A1, FloatRegister.F14, Register.A2, Register.A3]:
                        if history.isValid(reg):
                            argList.append((reg.name, history.read(reg)))
                        history.markBad(reg)
                    for s in (basicTypes.Stack(i) for i in range(0x10, 0x20, 4)):
                        if history.isValid(s):
                            argList.append(('stack_{:x}'.format(s.offset), history.read(s)))
                        else:
                            break
                
                marker = Symbol('returnValue_{:x}'.format((lineNum - 1)*4))
                currBlock.code.append((InstrResult.function, title, argList, marker))
                history.write(Register.V0, marker)
                history.write(FloatRegister.F0, marker)
            elif delayed[0] == InstrResult.end:
                if history.isValid(Register.V0):
                    returnValue = history.read(Register.V0)
                elif history.isValid(FloatRegister.F0):
                    returnValue = history.read(FloatRegister.F0)
                else:
                    returnValue = None
                currBlock.code.append((InstrResult.end, returnValue))
            delayed = None
    
    return mainCode, history, booleans


