import random

def randfloat(min, max):
    return random.random()*(max - min) + min

def diamondSquare(size, seed):
    random.seed(seed)
    map = [[0] * size for _ in range(size)]
    map[0][0] = 0
    map[0][size-1] = 0
    map[size-1][0] = 0
    map[size-1][size-1] = 0
    i = size - 1

    while i > 1:
        idt = i//2
        for x in range(idt, size, i):
            for y in range(idt, size, i):
                moy = (map[x - idt][y - idt] + map[x - idt][y + idt] + map[x + idt][y + idt] + map[x + idt][y - idt]) / 4
                map[x][y] = moy + (randfloat(-idt//2, idt//2) if i > 16 else 0)

        decalage = 0
        for x in range(0, size, idt):
            decalage = idt if decalage == 0 else 0
            for y in range(decalage, size, i):
                somme = 0
                n = 0
                if x >= idt:
                    somme += map[x - idt][y]
                    n += 1
                if x + idt < size:
                    somme += map[x + idt][y]
                    n += 1
                if y >= idt:
                    somme += map[x][y - idt]
                    n += 1
                if y + idt < size:
                    somme += map[x][y + idt]
                    n += 1
                map[x][y] = somme / n + (randfloat(-idt//2, idt//2) if i > 16 else 0)
        i = idt
    return map
