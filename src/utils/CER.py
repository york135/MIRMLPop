import numpy as np
from pypinyin import lazy_pinyin, Style

def CER(hypothesis: list, reference: list):
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1

            i -= 1
            j -= 1
        # elif ops_matrix[i_idx][j_idx] == 1:   # insert
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        # elif ops_matrix[i_idx][j_idx] == 2:   # delete
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    cer = wrong_cnt / len_ref

    # print("ref: %s" % " ".join(reference))
    # print("hyp: %s" % " ".join(hypothesis))
    # print(nb_map)
    # print("match_idx: %s" % str(match_idx))
    return cer, nb_map

def PER(hypothesis, reference):
    hypothesis_phoneme_initial = lazy_pinyin(hypothesis, style=Style.INITIALS, strict=False)
    hypothesis_phoneme_final = lazy_pinyin(hypothesis, style=Style.FINALS, strict=False)
    hypothesis_phoneme = []

    for i in range(len(hypothesis_phoneme_initial)):
        hypothesis_phoneme.append(hypothesis_phoneme_initial[i])
        hypothesis_phoneme.append(hypothesis_phoneme_final[i])


    target_phoneme_initial = lazy_pinyin(reference, style=Style.INITIALS, strict=False)
    target_phoneme_final = lazy_pinyin(reference, style=Style.FINALS, strict=False)
    target_phoneme = []

    for i in range(len(target_phoneme_initial)):
        target_phoneme.append(target_phoneme_initial[i])
        target_phoneme.append(target_phoneme_final[i])

    per, per_nb_map = CER(hypothesis=hypothesis_phoneme,
                        reference=target_phoneme)

    return per, per_nb_map

def Mandarin_SER(hypothesis, reference):
    hypothesis_syllable = lazy_pinyin(hypothesis)
    target_syllable = lazy_pinyin(reference)
    # print (hypothesis_syllable)
    # print (target_syllable)
    ser, ser_nb_map = CER(hypothesis=hypothesis_syllable,
                        reference=target_syllable)

    return ser, ser_nb_map

def SER_with_dict(hypothesis, reference, dict_data, reference_pron=None):
    
    hypothesis_syllable = []
    for i in range(len(hypothesis)):
        try:
            hypothesis_syllable.append(dict_data[hypothesis[i]][0])
        except:
            hypothesis_syllable.append(hypothesis[i])


    if reference_pron is not None:
        reference_syllable = reference_pron
    else:
        reference_syllable = []
        for i in range(len(reference)):
            try: 
                reference_syllable.append(dict_data[reference[i]][0])
            except:
                reference_syllable.append(reference[i])

    # print (hypothesis)
    # print (hypothesis_syllable)
    # print (reference_syllable)

    per, per_nb_map = CER(hypothesis=hypothesis_syllable,
                        reference=reference_syllable)

    return per, per_nb_map



def main():
    pass

if __name__ == "__main__":
    pass