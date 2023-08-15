from torch import IntTensor

from ao3_disco_ai.utils import convert_id_lists


def test_convert_id_lists():
    list_of_id_lists = [[1, 2, 3], [1, 2]]
    id_list, offsets = convert_id_lists(list_of_id_lists)
    assert (id_list == IntTensor([1, 2, 3, 1, 2])).all()
    assert (offsets == IntTensor([0, 3])).all()
