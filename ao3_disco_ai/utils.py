from typing import List

from torch import IntTensor


def convert_id_lists(list_of_id_lists: List[List[int]]) -> (IntTensor, IntTensor):
    """Convert a simple list of id_lists into a PyTorch input.

    Given a list where each entry corresponds to a list of sparse IDs, transform
    it into a id_list/offsets representation suitable for PyTorch.
    """
    id_list, offsets = [], []
    for single_id_list in list_of_id_lists:
        offsets.append(len(id_list))
        id_list.extend(single_id_list)
    id_list, offsets = IntTensor(id_list), IntTensor(offsets)
    return id_list, offsets
