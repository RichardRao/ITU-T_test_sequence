import numpy as np
from itertools import groupby

def binarizer(flag, stability:float, sample_rate:int):
    active = [flag.tolist()]
    pairs = count_same_pair(active)
    active = transform_to_binary_sequence(pairs, stability, sample_rate).flatten()
    segment = np.diff(np.append(0, active))
    segments = {"starts":[], "stops":[]}
    for i, ele in enumerate(segment):
        if ele == 1:
            segments["starts"].append(i)
        elif ele == -1:
            segments["stops"].append(i)
    if len(segments["starts"]) > len(segments["stops"]) and active[-1] == 1:
        segments["stops"].append(len(active)-1)
    
    return active, segments

def count_same_pair(nums):
    """Transform a list of 0 and 1 in a list of (value, num_consecutive_occurences).

    Args:
        nums (list): List of list containing the binary sequences.

    Returns:
        List of values and number consecutive occurences.

    Example:
        >>> nums = [[0,0,1,0]]
        >>> result = count_same_pair(nums)
        >>> print(result)
        [[[0, 2], [1, 1], [0, 1]]]

    """
    result = []
    for num in nums:
        result.append([[i, sum(1 for _ in group)] for i, group in groupby(num)])
    return result


def transform_to_binary_sequence(pairs, stability, sample_rate):
    """Transforms list of value and consecutive occurrences into a binary sequence with respect to stability

    Args:
        pairs (List): List of list of value and consecutive occurrences
        stability (Float): Minimal number of seconds to change from 0 to 1 or 1 to 0.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        np.tensor : The binary sequences.
    """

    batch_active = []
    for pair in pairs:
        active = []
        # Check for fully silent or fully voice sequence
        active, check = check_silence_or_voice(active, pair)
        if check:
            return active
        # Counter for every set of (value, num_consecutive_occ)
        i = 0
        # Do until every every sets as been used i.e until we have same sequence length as input length
        while i < len(pair):
            value, num_consecutive_occurrences = pair[i]
            # Counter for active set of (value, num_consecutive_occ) i.e (1,num_consecutive_occ)
            actived = 0
            # Counter for inactive set of (value, num_consecutive_occ) i.e (0,num_consecutive_occ)
            not_actived = 0
            # num_consecutive_occ <  int(stability * sample_rate) need to resolve instability
            if num_consecutive_occurrences < int(stability * sample_rate):
                # Resolve instability
                active, i = resolve_instability(
                    i, pair, stability, sample_rate, actived, not_actived, active
                )
            # num_consecutive_occ >  int(stability * sample_rate) we can already choose
            else:
                if value:
                    active.append(np.ones(pair[i][1]))
                else:
                    active.append(np.zeros(pair[i][1]))
                i += 1
        # Stack sequence to return a batch shaped tensor
        batch_active.append(np.hstack(active))
    batch_active = np.vstack(batch_active)
    return batch_active


def check_silence_or_voice(active, pair):
    """Check if sequence is fully silence or fully voice.

    Args:
        active (List) : List containing the binary sequence
        pair: (List): list of value and consecutive occurrences

    """
    value, num_consecutive_occurrences = pair[0]
    check = False
    if len(pair) == 1:
        check = True
        if value:
            active = np.ones(num_consecutive_occurrences)
        else:
            active = np.zeros(num_consecutive_occurrences)
    return active, check


def resolve_instability(i, pair, stability, sample_rate, actived, not_actived, active):
    """Resolve stability issue in input list of value and num_consecutive_occ

    Args:
        i (int): The index of the considered pair of value and num_consecutive_occ.
        pair (list) : Value and num_consecutive_occ.
        stability (float): Minimal number of seconds to change from 0 to 1 or 1 to 0.
        sample_rate (int): The sample rate of the waveform.
        actived (int) : Number of occurrences of the value 1.
        not_actived (int): Number of occurrences of the value 0.
        active (list) : The binary sequence.

    Returns:
        active (list) : The binary sequence.
         i (int): The index of the considered pair of value and num_consecutive_occ.
    """
    # Until we find stability count the number of samples active and inactive
    while i < len(pair) and pair[i][1] < int(stability * sample_rate):
        value, num_consecutive_occurrences = pair[i]
        if value:
            actived += num_consecutive_occurrences
            i += 1
        else:
            not_actived += num_consecutive_occurrences
            i += 1
    # If the length of unstable samples is smaller than the stability criteria and we are already in a state
    # then keep this state.
    if actived + not_actived < int(stability * sample_rate) and len(active) > 0:
        # Last value
        if active[-1][0] == 1:
            active.append(np.ones(actived + not_actived))
        else:
            active.append(np.zeros(actived + not_actived))
    # If the length of unstable samples is smaller than the stability criteria and but we have no state yet
    # then consider it silent
    elif actived + not_actived < int(stability * sample_rate) and len(active) == 0:
        active.append(np.zeros(actived + not_actived))
    # If the length of unstable samples is greater than the stability criteria then compare number of active
    # and inactive samples and choose.
    else:
        if actived > not_actived:
            active.append(np.ones(actived + not_actived))
        else:
            active.append(np.zeros(actived + not_actived))

    return active, i