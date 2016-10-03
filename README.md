# DAKompiler
a ~~simple~~ needlessly complicated mips decompiler 

## Basic usage
This decompiler doesn't target a particular version of MIPS, mostly because
I haven't implemented many opcodes that would have to differ by version. 

Python 3(.4ish?) required, no plans to implement a command line interface mostly 
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
the number is the address the binary data starts at, so that later you can use absolut addresses.

Access the main functionality through
```
decompileFunction(marioRam, bindings, address = 0x802c8504, args = ['A0 objA Object', 'A1 objB Object'])
```
where `address` is the absolute address your function of interest starts at, and `args` is a list of strings specifying which registers are used as arguments, the names that should be used, and their types, in a way I hope is clear. Other arguments DAKompiler discovers will be noted. 

## Planned features/fixes

### Major TODO items
- [ ] Recognize loop types
- [ ] Deal with JR and address tables
- [ ] Be more careful about types (maybe support C-style output)
- [ ] Deal with branching more intelligently, allow for `elif`
- [ ] Add 'likely' branches
- [ ] Add missing instructions (MULT, double precision math)
- [ ] Recognize flag and enum values

### Minor stuff
- [ ] Process branches better so merging doesn't take forever
- [ ] Simplify shift-based multiply-by-constant stuff
- [ ] Support for += etc.
- [ ] Other output options, like assembly offsets
- [ ] Display absolut branch destinations in mips instead of relative
- [ ] Handle functions that return a value more cleverly
- [ ] Make output formatting more extensible, all in its own `languageForm` file

### Ideas that sound cool but may take a while
- [ ] Identify memory areas that are constants and display them in the code
- [ ] Search for modifications to a given memory address
- [ ] Attempt to match unknown addresses with known struct types
- [ ] Smart recursive decompilation, maybe building a call tree