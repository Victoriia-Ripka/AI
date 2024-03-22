def logicalAND(x1, x2):
    if (x1 + x2 > 1.5):
        return 1
    else:
        return 0

def logicalOR(x1, x2):
    if (x1 + x2 > 0.5):
        if(x1 > x2):
            return x1
        else:
            return x2
    else:
        return 0

def logicalNOT(x1):
    if x1 * -1.5 > -1:
        return 1
    else:
        return 0

def logicalXOR(x1, x2):
    if x1 + x2 * -1 > 0.5:
        s1 = 1
    else:
        s1 = 0
    
    if x1 * -1 + x2 > 0.5:
        s2 = 1
    else:
        s2 = 0

    if s1 + s2 > 0.5:
        return 1
    else:
        return 0

a = 1
b = 0
print(f'Logical AND: {logicalAND(a, b)}')
print(f'Logical OR: {logicalOR(a, b)}')
print(f'Logical NOT: {logicalNOT(a)}')
print(f'Logical XOR: {logicalXOR(a, b)}')