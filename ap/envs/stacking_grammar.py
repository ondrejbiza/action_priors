import random


class StackingGrammar:

    def __init__(self, max_levels):

        self.max_levels = max_levels
        self.string = ["G"]
        # one block, two blocks next to each other, one brick, one small roof, one big roof
        self.terminals = ["1b", "2b", "1l", "1t", "2t"]
        # ground, wildcard, long object (2b, 1l, 2t), short object (1b, 1t)
        self.non_terminals = ["G", "W", "L", "S"]

        self.rules = {
            "G": {("1b", "S"), ("2b", "W"), ("1l", "W")},
            "W": {("S",), ("L",)},
            "S": {("1b", "S"), ("1b",), ("1t",)},
            "L": {("2b", "W"), ("1l", "W"), ("2b",), ("1l",), ("2t",)}
        }

        self.validate_rules_()

    def step(self):
        # create a random string with constraints
        i = 0

        while i < len(self.string):

            char = self.string[i]

            if char in self.non_terminals:
                rule = self.rules[char]

                if len(self.string) >= self.max_levels:
                    k = 0
                    while True:
                        outcome = random.choice(list(rule))
                        if self.does_not_increase_level_(outcome):
                            break
                        elif k > 1e+5:
                            raise ValueError("This is probably bad.")
                        k += 1
                else:
                    outcome = random.choice(list(rule))

                del self.string[i]

                for j in range(len(outcome)):
                    self.string.insert(i + j, outcome[j])

                i += len(outcome)
            else:
                i += 1

    def terminated(self):
        # check if string only contains terminals
        for char in self.string:
            if char in self.non_terminals:
                return False

        return True

    def get_string(self):
        # join array of symbols into string
        return "".join(self.string)

    def validate_rules_(self):
        # make sure rules make sense
        for key, value in self.rules.items():
            assert key in self.non_terminals
            for outcome in value:
                for char in outcome:
                    assert char in self.terminals or char in self.non_terminals

    def does_not_increase_level_(self, outcome):
        # L -> 1lW does increase height
        # L -> 1l does not
        # make sure the current string does not increase height
        for char in outcome:
            if char not in self.terminals:
                if len(outcome) > 1:
                    # the outcome contains a non-terminal and the length of the outcome is greater than one
                    return False

        return True


def generate_strings_height_3_roof():
    # generate all strings of height three with a roof on top
    # instead of actually enumerating everything, I generate a very high number of random strings ...
    strings = set()

    for i in range(int(1e+5)):

        if i % 100 and i > 0:
            print("step {:d}, {:d} unique strings".format(i, len(strings)))

        g = StackingGrammar(3)

        while not g.terminated():
            g.step()

        strings.add(g.get_string())

    strings = list(strings)

    for i in range(len(strings)):

        strings[i] = strings[i].replace("t", "r")

    roof_strings = []
    for string in strings:
        if string[-2:] == "1r" or string[-2:] == "2r":
            roof_strings.append(string)

    roof_strings = list(sorted(roof_strings))
    return roof_strings


def count_objects(string):
    # count the number of objects used in a string
    count = 0

    for i in range(len(string) // 2):

        chars = string[i * 2: (i + 1) * 2]

        if chars == "2b":
            count += 2
        elif chars in ["1b", "1l", "1r", "2r"]:
            count += 1
        else:
            raise ValueError("bad")

    return count


def cmp(x, y):
    # this is used for sorting the goal strings
    assert len(x) % 2 == 0 and len(y) % 2 == 0
    terminals = ["1b", "2b", "1l", "1r", "2r"]

    if len(x) == len(y):
        for i in range(len(x) // 2):
            s1 = x[i * 2: (i + 1) * 2]
            s2 = y[i * 2: (i + 1) * 2]

            if s1 == s2:
                continue

            i1 = terminals.index(s1)
            i2 = terminals.index(s2)

            if i1 <= i2:
                return -1
            else:
                return 1

        return 0
    else:
        if len(x) < len(y):
            return -1
        else:
            return 1
