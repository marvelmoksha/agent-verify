import json

import pytest

from benchmarks.dev_subset import load_dev_ids, save_dev_ids, select_dev_ids

HE = [f"humaneval/{i}" for i in range(164)]
MB = [f"mbpp/{i}" for i in range(11, 511)]


def test_deterministic_and_right_size():
    a = select_dev_ids(HE, MB, 25, 42)
    b = select_dev_ids(list(reversed(HE)), list(reversed(MB)), 25, 42)
    assert a == b
    assert len(a) == 50 == len(set(a))
    assert sum(i.startswith("humaneval/") for i in a) == 25


def test_different_seed_changes_selection():
    assert select_dev_ids(HE, MB, 25, 1) != select_dev_ids(HE, MB, 25, 2)


def test_save_refuses_overwrite(tmp_path):
    p = tmp_path / "dev_ids.json"
    ids = select_dev_ids(HE, MB)
    save_dev_ids(ids, 42, 25, p)
    with pytest.raises(FileExistsError):
        save_dev_ids(ids, 42, 25, p)
    assert load_dev_ids(p) == ids


def test_edit_is_detected(tmp_path):
    p = tmp_path / "dev_ids.json"
    save_dev_ids(select_dev_ids(HE, MB), 42, 25, p)
    data = json.loads(p.read_text())
    data["ids"][0] = "humaneval/999"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_dev_ids(p)
