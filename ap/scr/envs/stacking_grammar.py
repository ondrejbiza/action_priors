from ...envs.stacking_grammar import StackingGrammar


strings = set()


for i in range(int(1e+5)):

    if i % 100 and i > 0:
        print("step {:d}, {:d} unique strings".format(i, len(strings)))

    g = StackingGrammar(3)

    while not g.terminated():
        g.step()

    strings.add(g.get_string())

print(sorted(strings, key=len))
