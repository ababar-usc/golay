import numpy as np
"""
1. gen stab (or, equiv, parity check) matrix
2. find all ops that commute with all rows of stab matrix
3. select logical ops
4. fix gauge ops
"""

golay_gen_23 = [0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1]

# 1: non-rr first 2 cols drop (parity bits)
# 2: non-rr drop 2 cols with a single 1 (col 12,13 of Matrix form: [M|reversed_I])
# 3: rr drop first 2 parity cols (col 13,14 of Matrix form: [I|P])
punctured_golay_stabs_1 = [
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]

punctured_golay_stabs_2 = [
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

punctured_golay_stabs_3 = [
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]

def gen_all_stabs(stabs):
    all_stabs = set(stabs)
    for stab1 in stabs:
        new_stabs = set()
        for stab2 in all_stabs:
            new_stabs.add(stab1 ^ stab2)
        all_stabs |= new_stabs
    return all_stabs

def convert_stab_mat_to_bin_set(stabs):
    bin_stabs = set()
    for stab in stabs:
        stab_str = ''.join(str(num) for num in stab)
        bin_stabs.add(int(stab_str, 2))
    return bin_stabs

bin_stabs_1 = convert_stab_mat_to_bin_set(punctured_golay_stabs_1)
bin_stabs_2 = convert_stab_mat_to_bin_set(punctured_golay_stabs_2)
bin_stabs_3 = convert_stab_mat_to_bin_set(punctured_golay_stabs_3)
# print(len(bin_stabs_1),len(bin_stabs_2),len(bin_stabs_3))

all_bin_stabs_1 = gen_all_stabs(bin_stabs_1)
all_bin_stabs_2 = gen_all_stabs(bin_stabs_2)
all_bin_stabs_3 = gen_all_stabs(bin_stabs_3)
# print(len(all_bin_stabs_1),len(all_bin_stabs_2),len(all_bin_stabs_3))
# print(0 in all_bin_stabs_1, 0 in all_bin_stabs_2, 0 in all_bin_stabs_3)

print(all_bin_stabs_1 == all_bin_stabs_2)
print(all_bin_stabs_2 == all_bin_stabs_3)
print(all_bin_stabs_1 == all_bin_stabs_3)
