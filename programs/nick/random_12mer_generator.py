

import random
import string




#"A": 0.0,
#"C": 0.0,
#"D": 0.0,
#"E": 0.0,
#"F": 1.0,
#"G": 0.0,
#"H": 0.0,
#"I": 0.0,
#"K": 0.0,
#"L": 0.0,
#"M": 0.0,
#"N": 0.0,
#"P": 0.0,
#"Q": 0.0,
#"R": 0.0,
#"S": 0.0,
#"T": 0.0,
#"V": 0.0,
#"W": 1.0,
#"Y": 1.0



# random = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])


amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

with open('random_output.txt', 'w') as writeFile:
    for i in range(10000):
        random_12mer = ''.join([random.choice(amino_acids) for n in range(12)])
        writeFile.write(random_12mer + '\n')
