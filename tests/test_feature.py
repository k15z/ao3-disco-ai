from ao3_disco_ai.feature import parse_chapters

def test_parse_chapters():
    assert parse_chapters("1/1") == [1, 1, 1]
    assert parse_chapters("3/5") == [3, 5, 0]
    assert parse_chapters("3/?") == [3, 0, 0]
