"""The one JSON Lines reader and writer: blank lines, malformed lines, separators inside strings, and the bytes written."""
import io
import json

import pytest

from rewardhack import jsonl
from rewardhack.json_objects import unique_object

ROWS = [dict(id=1, text='plain'), dict(id=2, text='é and ∀ x, x = x'), dict(id=3, text='line separator\x85next line'), dict(id=4, nested=dict(a=[1, None, True]))]


def test_blank_lines_are_skipped_and_rows_keep_their_order(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_text('\n' + jsonl.dumps(ROWS[:2]) + '   \n\t\n' + jsonl.dumps(ROWS[2:]) + '\n')
    assert list(jsonl.read(path)) == ROWS


def test_a_malformed_line_names_the_file_and_its_line_number(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_text(jsonl.dumps(ROWS[:1]) + '\n{"id": \n')
    with pytest.raises(jsonl.MalformedLine, match=r'line 3 of .*rows\.jsonl is not JSON'):
        list(jsonl.read(path))
    with pytest.raises(ValueError, match='line 2 of the probe output'):
        list(jsonl.loads('{}\nnot json\n', 'the probe output'))


def test_rows_before_a_malformed_line_are_yielded_first(tmp_path):
    rows = jsonl.loads(jsonl.dumps(ROWS[:2]) + '{\n', 'source')
    assert [next(rows), next(rows)] == ROWS[:2]
    with pytest.raises(jsonl.MalformedLine):
        next(rows)


def test_line_and_paragraph_separators_inside_a_string_do_not_split_a_row(tmp_path):
    text = jsonl.dumps(ROWS)
    assert ' ' in text and '\x85' in text  # written unescaped with ensure_ascii=False
    assert len(text.splitlines()) > len(ROWS)  # which str.splitlines would have cut
    assert list(jsonl.loads(text, 'memory')) == ROWS
    path = tmp_path/'rows.jsonl'
    path.write_text(text)
    assert list(jsonl.read(path)) == ROWS


def test_windows_and_old_mac_line_ends_are_read(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_bytes(b'{"a": 1}\r\n{"a": 2}\r{"a": 3}\n')
    assert [row['a'] for row in jsonl.read(path)] == [1, 2, 3]
    assert [row['a'] for row in jsonl.loads('{"a": 1}\r\n{"a": 2}\r{"a": 3}', 'memory')] == [1, 2, 3]


def test_the_object_pairs_hook_sees_every_object_and_its_error_passes_through():
    with pytest.raises(ValueError, match='duplicate') as caught:
        list(jsonl.loads('{"a": 1}\n{"a": 1, "a": 2}\n', 'evidence', object_pairs_hook=unique_object))
    assert not isinstance(caught.value, jsonl.MalformedLine)


@pytest.mark.parametrize('ensure_ascii', [False, True])
def test_written_bytes_equal_the_expression_each_writer_used_before(ensure_ascii):
    expected = ''.join(json.dumps(row, ensure_ascii=ensure_ascii)+'\n' for row in ROWS)
    assert jsonl.dumps(ROWS, ensure_ascii) == expected
    one, many = io.StringIO(), io.StringIO()
    for row in ROWS:
        jsonl.write(one, row, ensure_ascii=ensure_ascii)
    jsonl.write_rows(many, iter(ROWS), ensure_ascii=ensure_ascii)
    assert one.getvalue() == many.getvalue() == expected


def test_the_default_writes_unicode_unescaped():
    sink = io.StringIO()
    jsonl.write(sink, dict(text='∀'))
    assert sink.getvalue() == '{"text": "∀"}\n'


def test_each_write_is_flushed():
    class Sink(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()
    sink = Sink()
    jsonl.write(sink, ROWS[0])
    jsonl.write_rows(sink, ROWS)
    assert sink.flushes == 2
