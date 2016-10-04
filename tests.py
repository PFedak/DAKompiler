import unittest
import basicTypes
import symbolify

Context = symbolify.Context


class TestContexts(unittest.TestCase):
    def test_basicBranching(self):
        base = symbolify.Branch().branchOff(0, True)
        left = Context([base.branchOff(1, True)])
        right = Context([base.branchOff(1, False)])
        self.assertFalse(left.isCompatibleWith(right))
        imp, rel = left.implies(Context([base]))
        self.assertTrue(imp)
        self.assertEqual(len(rel.cnf), 1)
        self.assertEqual(len(rel.cnf[0]), 1)
        self.assertEqual(rel.cnf[0][1], True)

    def test_completeMerge(self):
        for L in range(3, 8):  # currently takes ~2.3 seconds on my machine
            base = []
            for i in range(1 << L):
                temp = symbolify.Branch()
                for j in range(L):
                    temp = temp.branchOff(j, i & (1 << j))
                base.append(temp)
            self.assertTrue(Context(base).isTrivial())

    @unittest.skip('takes a while')
    def test_bigMerge(self):
        # currently takes ~2.3 seconds on my machine
        base = []
        L = 9
        for i in range(1 << L):
            temp = symbolify.Branch()
            for j in range(L):
                temp = temp.branchOff(j, i & (1 << j))
            base.append(temp)
        self.assertTrue(Context(base).isTrivial())


build = symbolify.Expression.build
Lit = symbolify.Literal
Sym = symbolify.Symbol


class TestExpressions(unittest.TestCase):
    def test_adding(self):
        three = build('+', Lit(1), Lit(2))
        self.assertEqual(three.value, 3)
        bar = build('+', Sym('bar'), Lit(0))
        self.assertTrue(isinstance(bar, Sym))
        self.assertEqual('{}'.format(bar), 'bar')
        # symbols are sent to the left
        together = build('+', three, bar)
        self.assertEqual('{}'.format(together), 'bar + 3')
        self.assertEqual('{}'.format(build('+', together, together)), 'bar + bar + 6')

    def test_hexFormat(self):
        self.assertEqual('{:h}'.format(Lit(255)), '0xff')
        self.assertEqual('{}'.format(build('|', Sym('x'), Lit(21))), 'x | 0x15')

    @unittest.skip('still figuring out how this should work')
    def test_flag(self):
        testFlags = basicTypes.Flag(basicTypes.short, {0: 'zero', 1: 'one', 2: 'two'})
        flagVar = Symbol('flags', testFlags)
        self.assertEqual('{}'.format(build('&', 'flags', Lit(2))), 'flags[one]')
        self.assertEqual('{}'.format(build('&', 'flags', Lit(5))), 'flags[zero] or flags[two]')


Var = basicTypes.Variable


class TestStructLookup(unittest.TestCase):
    def setUp(self):
        bindings = {'structs': {
            'testStruct': basicTypes.StructType('testStruct', 4 + 4 + 4 + 8 + 4, {
                0: Var('zero', basicTypes.word),
                4: Var('sub', 'subStruct'),
                # 8: word (missing)
                0xc: Var('array', basicTypes.Array(basicTypes.short, 4)),
                0x14: Var('smallNumber', 'testEnum'),

            }),
            'subStruct': basicTypes.StructType('subStruct', 4, {
                0: Var('a', basicTypes.short),
                2: Var('b', basicTypes.byte),
                3: Var('c', basicTypes.byte),
            })
        },
            'enums': {
                'testEnum': basicTypes.EnumType('testEnum', basicTypes.word, {
                    0: 'zero',
                    1: 'one',
                    2: 'two'
                })
            }
        }
        self.foo = Sym('foo', 'testStruct')
        self.history = symbolify.VariableHistory(bindings)
        self.ptr = Sym('structPointer', basicTypes.Pointer('testStruct', 'bar'))

    def test_sizes(self):
        gs = self.history.getSize
        self.assertEqual(gs(basicTypes.ushort), 2)
        self.assertEqual(gs(basicTypes.Flag(basicTypes.byte, {})), 1)
        self.assertEqual(gs(self.ptr.type), 4)
        self.assertEqual(gs('testEnum'), 4)
        self.assertEqual(gs(basicTypes.Array(basicTypes.short, 5)), 10)
        self.assertEqual(gs(basicTypes.Array(self.ptr.type, 3)), 12)
        self.assertEqual(gs('testStruct'), 24)

    def test_member(self):
        self.assertEqual(self.history.lookupAddress(basicTypes.word, self.foo).name, 'foo.zero')

    def test_subMember(self):
        self.assertEqual(self.history.lookupAddress(basicTypes.short, build('+', self.foo, Lit(4))).name, 'foo.sub.a')
        self.assertEqual(self.history.lookupAddress(basicTypes.byte, build('+', self.foo, Lit(7))).name, 'foo.sub.c')

    def test_array(self):
        self.assertEqual(self.history.lookupAddress(basicTypes.short, build('+', self.foo, Lit(0xc))).name,
                         'foo.array[0]')
        self.assertEqual(self.history.lookupAddress(basicTypes.short, build('+', self.foo, Lit(0xe))).name,
                         'foo.array[1]')

    @unittest.expectedFailure
    def test_varArray(self):
        address = Expression('+', [self.foo, build('<<', Sym('bar'), Lit(1)), Lit(8)])
        self.assertEqual(self.history.lookupAddress(basicTypes.short, address), 'foo.array[bar]')

    def test_unknown(self):
        self.assertEqual(self.history.lookupAddress(basicTypes.short, build('+', self.foo, Lit(8))).name, 'foo.h_0x8')

    def test_tooFar(self):
        self.assertRaises(Exception, self.history.lookupAddress, basicTypes.word, build('+', self.foo, Lit(0x20)))

    def test_pointer(self):
        indirect = self.history.lookupAddress(basicTypes.word, self.ptr)
        self.assertEqual(indirect.name, 'bar')
        self.assertEqual(indirect.type, 'testStruct')


if __name__ == '__main__':
    unittest.main()
