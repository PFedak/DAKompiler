from symbolify import InstrResult as IR, Context

indent = ' '*4

def dummy(*args):
    return str(args)

def renderReg(state):
    return '{0.name} = {0.value}'.format(state) if state.explicit else None

def renderFunc(title, args, val):
    return '{} = {}({})'.format(val, title, 
        ', '.join('{} = {}'.format(r,v) if r else format(v) for r,v in args))

def renderWrite(value, target):
    return '{} = {}'.format(target, value)

def renderReturn(value = None):
    return 'return {}'.format(value) if value else None

renderList = {
    IR.register : renderReg,
    IR.write    : renderWrite,
    IR.function : renderFunc,
    IR.end      : renderReturn,
    IR.unhandled: str
    }

def renderFunctionToPython(name, codeTree, history, booleans):
    text = ['def {}({}):'.format(name,', '.join(history.states[arg][0].value.name for arg in history.argList))]
    text.extend(renderToPython(codeTree,booleans,1))
    return text

def renderToPython(codeTree, booleans, level = 0):
    text = []
    for line in codeTree.code:
        result = renderList[line[0]](*line[1:])
        if result:
            text.append((indent*level)+result)
    previousRelative = Context()
    for block in codeTree.children:
        if block.relative.isTrivial():
            newLevel = level
            prefix = None
        else:
            newLevel = level + 1
            keyword = '{}if {}:' if block.relative.isCompatibleWith(previousRelative) else '{}elif {}:'
            prefix = keyword.format(indent*level, 
                    ' or '.join(
                        ' and '.join(
                            ('{}' if val else '{:!}').format(booleans[ch]) for ch, val in br.items()
                        )
                    for br in block.relative.cnf)
                )
        inner = renderToPython(block, booleans, newLevel)
        if inner:
            if prefix:
                text.append(prefix)
            text.extend(inner)
            previousRelative = block.relative
    return text

