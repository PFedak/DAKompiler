import struct
import itertools
from collections import defaultdict
from functools import reduce
import basicTypes
import algebra
from instruction import *

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
            return algebra.Expression.build(op, history.read(instr.sourceReg, basicTypes.word), algebra.Literal(instr.immediate))
        elif fmt == 'S+I':
            return algebra.Expression.build(op, history.read(instr.sourceReg, basicTypes.word), algebra.Literal(extend(instr.immediate)))
        elif fmt == 'f(I)':
            return algebra.Literal(op(instr.immediate))
        elif fmt == 'F+F':
            return algebra.Expression.build(op, history.read(instr.fs, basicTypes.single), 
                                        history.read(instr.ft, basicTypes.single), flop=True)
        elif fmt == '@F':
            if op == 'id':
                return history.read(instr.fs)
            else:
                return algebra.Expression.build('@', op, history.read(instr.fs))
        elif fmt == 'T<<A':
            return algebra.Expression.build(op, history.read(instr.targetReg), algebra.Literal(instr.shift))
        elif fmt == 'S+T':
            return algebra.Expression.build(op, history.read(instr.sourceReg, basicTypes.word),
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
            address = algebra.Expression.build('+', history.read(instr.sourceReg, basicTypes.address), 
                                            algebra.Literal(extend(instr.immediate)))
            value = history.lookupAddress(datatype, address)
        return InstrResult.register, history.write(instr.targetReg, value)
    return foo

def storeMemory(datatype):
    def foo(instr, history):
        if instr.sourceReg == Register.SP:
            return InstrResult.register, history.write(basicTypes.Stack(extend(instr.immediate)),
                                                        history.read(instr.targetReg, datatype)) 
        else:
            dest = history.lookupAddress(datatype, algebra.Expression.build('+', 
                                                        history.read(instr.sourceReg, basicTypes.address), 
                                                        algebra.Literal(extend(instr.immediate))))
            return InstrResult.write, history.read(instr.targetReg, dest.type), dest
    return foo

def branchMaker(comp, withZero, likely = False):
    def doBranch(instr, history):
        if instr.opcode == MainOp.BEQ and instr.sourceReg == instr.targetReg:
            return InstrResult.jump, None, extend(instr.immediate)
        return (InstrResult.likely if likely else InstrResult.branch, 
                    algebra.Expression.build(comp, 
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
    RegOp.JALR: lambda instr, history: (InstrResult.function, '{}'.format(history.read(instr.sourceReg, basicTypes.address))), 
    RegOp.ADD: assignReg('D','S+T','+'),
    RegOp.ADDU: assignReg('D','S+T','+'),
    RegOp.SUB: assignReg('D','S+T','-'),
    RegOp.SUBU: assignReg('D','S+T','-'),
    RegOp.DIV: assignReg('D','S+T','/'),
    RegOp.AND: assignReg('D','S+T','&'),
    RegOp.OR: assignReg('D','S+T','|'),
    RegOp.XOR: assignReg('D','S+T','^'),
    RegOp.NOR: assignReg('D', 'S+T', 'NOR'),
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

    def givenNot(self, badList):
        drop = []
        for br in badList:
            if len(br) == 1:
                drop.extend(br)
        return self.without(drop)

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
                        prime.add(lower)
        if len(prime) == 1:
            self.cnf = list(prime)
        else:
            prime = sorted(prime, key = lambda x:(min(x), len(x)))
            self.cnf = []
            for br in prime:
                self.cnf.append(br.givenNot(self.cnf))

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

    def processElif(self, badList):
        """What parts of this matter, given that other is NOT true"""
        dropped = [br.without(badList) for br in self.cnf]
        for br in dropped:
            if len(br) == 1:
                badList.extend(br)
        return Context(dropped)

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
        self.write(Register.R0, algebra.Literal(0))
        self.write(Register.SP, algebra.Symbol('SP'))
        self.write(Register.RA, algebra.Symbol('RA'))
        self.write('CC', algebra.Symbol('bad_CC'))
        for reg, name, fmt in args:
            showName = name if name else VariableHistory.getName(arg)
            self.argList.append(reg)
            self.write(reg, algebra.Symbol(showName, fmt))

    def read(self, var, fmt = basicTypes.unknown):
        """Retrive (an appropriate representation of) the value in a register and track its usage
            
            var should be a register or Stack() object

            Depending on the expected format, the stored value may be altered substantially

        """
        if var == Register.R0:  #zero is zero, shouldn't remember type info
            return algebra.Literal(0)
        if var in self.states:
            uncertain = False
            for st in reversed(self.states[var]):
                if self.now.implies(st.context)[0]: # this state definitely occurred
                    if uncertain:
                        st.explicit = True
                        break
                    else:
                        if isinstance(st.value, algebra.Literal):
                            if isinstance(fmt, basicTypes.EnumType):
                                st.value = self.getEnumValue(fmt, st.value.value)

                        elif basicTypes.isAddressable(fmt):
                            st.value = self.lookupAddress(fmt, st.value)
                        elif st.value.type == basicTypes.unknown:
                            st.value.type = fmt

                        return st.value
                elif self.now.isCompatibleWith(st.context):
                    st.explicit = True
                    uncertain = True
            return algebra.Symbol(VariableHistory.getName(var), fmt)
        else:
            symName = VariableHistory.getName(var)
            if VariableHistory.couldBeArg(var): 
                self.argList.append(var)
                symName = 'arg_' + symName
            self.states[var].append(VariableState(self.getName(var), algebra.Symbol(symName, fmt), self.now))
            return self.states[var][-1].value
        

    def write(self, var, value):
        self.states[var].append(VariableState(self.getName(var), value, self.now))
        return self.states[var][-1]

    def markBad(self, var):
        self.write(var, algebra.Symbol('bad_%s' % VariableHistory.getName(var), basicTypes.bad))    

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
        """Find data of an appropriate format at a (possibly symbolic) address
        
            fmt can influence the results
                lookupAddress(single, address_of_v)  ->   v.x
                lookupAddress(Vector, address_of_v)  ->   v

            if fmt is unknown (or no match is found), ?????
        """
        if isinstance(address, algebra.Literal) and address.value in self.bindings['globals']:
            base = algebra.Symbol(*self.bindings['globals'][address.value])
            if basicTypes.isAddressable(base.type):
                return self.subLookup(fmt, base.name, base.type, 0)
            else:
                return base     

        base = None
        memOffset = 0
        others = []     
        if isinstance(address, algebra.Literal):
            memOffset = address.value       
        elif isinstance(address, algebra.Symbol):
            if isinstance(address.type, basicTypes.Pointer):
                return algebra.Symbol(address.type.target if address.type.target else address.name, address.type.pointedType)
            if basicTypes.isAddressable(address.type):
                base = address
        elif isinstance(address, algebra.Expression) and address.op == '+':
            memOffset = address.constant.value if address.constant else 0
            for term in address.args:
                if basicTypes.isAddressable(term.type):
                    if base:
                        raise Exception('adding structs')
                    base = term
                else:
                    others.append(term)

        if not base:
            #check for trig lookup
            if fmt == basicTypes.single and memOffset in self.bindings['trigtables']:
                try:
                    angle = others[0].args[0].args[0]
                    return algebra.Symbol('{}Table({})'.format(self.bindings['trigtables'][memOffset], angle))
                except:
                    pass
            pair = self.relativeToGlobals(memOffset)
            if pair:
                return self.subLookup(fmt, pair[0].name, pair[0].type, pair[1], others)

            # no idea what we are looking at, process it anyway
            return algebra.Symbol('{}({:h})'.format(fmt, address), fmt)
        if memOffset >= self.getSize(base.type):
            raise Exception('trying to look up address %#x in %s (only %#x bytes)' % (memOffset, base.type, self.getSize(base.type)))

        # determine which member struct we are trying to access
        return self.subLookup(fmt, base.name, base.type, memOffset, others)

    def relativeToGlobals(self, offset):
        try:
            bestOffset = max(x for x in self.bindings['globals'] if x <= offset)
        except ValueError:
            return None     # nothing less than this value

        base = algebra.Symbol(*self.bindings['globals'][bestOffset])
        relOffset = offset - bestOffset
        if relOffset >= self.getSize(base.type):
            return None
        else:
            return base, relOffset

    def subLookup(self, fmt, prefix, superType, address, others = []):
        """Recursively find data at the given address from the start of a type"""
        if isinstance(superType, basicTypes.Array):
            index = algebra.Literal(address//self.getSize(superType.pointedType))
            canIndex = True
            if others:
                for o in others:
                    try:
                        if o.op == '*' and o.constant.value == self.getSize(superType.pointedType):
                            index = algebra.Expression.build('+', index, algebra.Expression.arithmeticMerge('*', o.args))
                        else:
                            canIndex = False
                            break 
                    except:
                        canIndex = False
                        break 
            if canIndex:
                return algebra.Symbol('{}[{}]'.format(prefix, index), superType.pointedType)
            else:
                offset = algebra.Expression.arithmeticMerge('+', others + [algebra.Literal(address)])
                return algebra.Symbol('{}({} + {:h})'.format(fmt, prefix, offset), superType.pointedType)
        if isinstance(superType, basicTypes.Pointer):
            return algebra.Symbol(superType.target if superType.target else prefix, superType.pointedType)
        if isinstance(superType, str):
            try:
                members = self.bindings['structs'][superType].members
                bestOffset = max(x for x in members if x <= address)
                base = algebra.Symbol(*members[bestOffset])
                if address < bestOffset + self.getSize(base.type):
                    if basicTypes.isAddressable(base.type):
                        return self.subLookup(fmt, '{}.{}'.format(prefix, base), base.type, address - bestOffset, others)
                    if not others:
                        #TODO account for reading the lower short of a word, etc.
                        return algebra.Symbol('{}.{}'.format(prefix, base), base.type)
            except:
                pass    #unknown struct, nothing in max(), went too far
        # nothing matched
        if others:
            return algebra.Symbol('{}({} + {:h})'.format(fmt, prefix, algebra.Expression.build('+', algebra.Expression('+',others), algebra.Literal(address))), fmt)
        else:
            return algebra.Symbol('{}.{}_{:#x}'.format(prefix, basicTypes.getCode(fmt), address), fmt)

    def getEnumValue(self, fmt, val):
        try:
            subname = self.bindings['enums'][fmt.enum].values[val]
        except:
            subname = '_{:#x}'.format(val)
        return algebra.Symbol('{}.{}'.format(fmt.enum, subname), fmt)


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
        self.elifAccumulator = []
        self.elseRelative = None

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
            # different set of active branches, start a new block of code
            newContext = Context([b for b in branchList if 0 <= b.line <= lineNum], lineNum)
            newParent = currBlock
            while True:
                imp, rel = newContext.implies(newParent.context)
                if imp:
                    break
                else:
                    newParent = newParent.parent
            currBlock = CodeBlock(newContext, newParent, rel)
            # continue an elif chain, or start a new one
            if newParent.children and not rel.isCompatibleWith(newParent.children[-1].relative):
                currBlock.elseRelative = rel.processElif(newParent.elifAccumulator)
            else:
                newParent.elifAccumulator = [list(br)[0] for br in rel.cnf if len(br) == 1]
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
                    try:
                        title = 'fn%06x' % funcCall
                    except:
                        title = funcCall
                    for reg in [Register.A0, FloatRegister.F12, Register.A1, FloatRegister.F14, Register.A2, Register.A3]:
                        if history.isValid(reg):
                            argList.append((reg.name, history.read(reg)))
                        history.markBad(reg)
                    for s in (basicTypes.Stack(i) for i in range(0x10, 0x28, 4)):
                        if history.isValid(s):
                            argList.append(('stack_{:x}'.format(s.offset), history.read(s)))
                            history.markBad(s)
                        else:
                            break
                
                marker = algebra.Symbol('returnValue_{:x}'.format((lineNum - 1)*4))
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


