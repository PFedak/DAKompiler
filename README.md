# DAKompiler
a ~~simple~~ needlessly complicated mips decompiler 

## Basic usage
This decompiler doesn't target a particular version of MIPS, mostly because
I haven't implemented many opcodes that would have to differ by version. Nor does it try to accurately decompile arbitrary MIPS; I'm targeting code that was written by a non-very-aggressive compiler, which seems to be enough for most game logic in Super Mario 64.

Python >= 3.4 required, no plans to implement a command line interface mostly 
because I've found this to be a rather interactive process most of the time.

Load a 'bindings' object from a ram map file via
```
bindings = loadBindings('sm64 ram map.txt', 'J')
```
where the region code is required (function and global variable addresses will generally differ by region).
The included file is far from complete.

You'll also need binary data, either a dump of the game's RAM at some point or the ROM itself (although the latter may eventually cause problems because absolute addresses will be wrong)
```
marioRam = RAMSnapshot('marioRam', 0x80000000)
```
the number is the address the binary data starts at, so that later you can use absolute addresses.

Access the main functionality through
```
decompileFunction(marioRam, bindings, address = 0x8027D14C, args = ['A0 obj Object'])
```
where `address` is the absolute address your function of interest starts at, and `args` is a list of strings specifying which registers are used as arguments, the names that should be used, and their types, in a way I hope is clear. Other arguments DAKompiler discovers will be noted. 

With any luck, you'll see something along the lines of
```
def fn8027d14c(obj):
    if obj._0x18 == ubyte(word(0X8032CF90) + 0x14):
        if obj.transform != 0:
            returnValue_74 = fn80379f60(A0 = (short(0X8033A770) << 6) + 0x8033a7b8, A1 = obj.transform, A2 = (short(0X8033A770) << 6) + 0x8033a778)
        else:
            if (obj.gfxFlags & 0x4) != 0:
                returnValue_c8 = fn80379798(A0 = (short(0X8033A770) << 6) + 0x8033a7b8, A1 = (short(0X8033A770) << 6) + 0x8033a778, A2 = obj + 32, A3 = short(word(0X8032CF9C) + 0x38))
            else:
                returnValue_e4 = fn80379440(A0 = SP + -64, A1 = obj + 32, A2 = obj + 26)
                returnValue_10c = fn80379f60(A0 = (short(0X8033A770) << 6) + 0x8033a7b8, A1 = SP + -64, A2 = (short(0X8033A770) << 6) + 0x8033a778)
        returnValue_13c = fn8037a29c(A0 = (short(0X8033A770) << 6) + 0x8033a7b8, A1 = (short(0X8033A770) << 6) + 0x8033a7b8, A2 = obj + 44)
        short(0X8033A770) = ((short(0X8033A770) + 1) << 16) >a 16
        obj.transform = ((((short(0X8033A770) + 1) << 16) >a 16) << 6) + 0x8033a778
        obj.posOffset.x = single((short(0X8033A770) << 0x6) + 0x8033a7a8)
        obj.posOffset.y = single((short(0X8033A770) << 0x6) + 0x8033a7ac)
        obj.posOffset.z = single((short(0X8033A770) << 0x6) + 0x8033a7b0)
        if obj._0x3c != 0:
            returnValue_1f8 = fn8027c988(A0 = obj + 56, A1 = (obj.gfxFlags & 0x20) < 0)
        returnValue_218 = fn8027cf68(A0 = obj, A1 = (short(0X8033A770) << 6) + 0x8033a778)
        if returnValue_218 != 0:
            returnValue_228 = fn8027897c(A0 = 64)
            returnValue_24c = fn8037a434(A0 = returnValue_228, A1 = (short(0X8033A770) << 6) + 0x8033a778)
            word((short(0X8033A770) << 0x2) + 0x8033af78) = returnValue_228
            if obj._0x14 != 0:
                word(0X8032CFA0) = obj
                word(obj._0x14 + 0xc) = obj
                returnValue_29c = fn8027d8f8(A0 = obj._0x14)
                word(obj._0x14 + 0xc) = 0
                word(0X8032CFA0) = 0
            if obj.gfxChild != 0:
                returnValue_2cc = fn8027d8f8(A0 = obj.gfxChild)
        short(0X8033A770) = short(0X8033A770) + -1
        byte(0X8033B008) = 0
        obj.transform = 0
    return V0
```
which, y'know, could be *less* readable.

## Planned features/fixes

### Major TODO items
- [ ] Recognize loop types
- [ ] Deal with JR and address tables
- [ ] Be more careful about types (maybe support C-style output)
- [x] Deal with branching more intelligently, allow for `elif`
- [ ] Add 'likely' branches
- [ ] Add missing instructions (MULT, double precision math)
- [ ] Recognize flag and enum values

### Minor stuff
- [ ] Process branches better so merging doesn't take forever
- [ ] Simplify shift-based multiply-by-constant stuff
- [ ] Support for += etc.
- [ ] Other output options, like assembly offsets
- [ ] Display absolute branch destinations in mips instead of relative
- [ ] Handle functions that return a value more cleverly
- [ ] Make output formatting more extensible, all in its own `languageForm` file

### Ideas that sound cool but may take a while
- [ ] Identify memory areas that are constants and display them in the code
- [ ] Search for modifications to a given memory address
- [ ] Attempt to match unknown addresses with known struct types
- [ ] Smart recursive decompilation, maybe building a call tree
