"""The suite's own GPU selection: physical indices, inside the allocation."""

from _support import free_gpus

ROWS = [("0", 15000), ("1", 45000), ("2", 500), ("3", 80000), ("4", 20000), ("5", 68000)]


def test_without_an_allocation_every_free_gpu_is_a_candidate():
    assert free_gpus(12000, visible=None, rows=ROWS) == ["0", "1", "3", "4", "5"]


def test_an_allocation_bounds_and_orders_the_candidates():
    assert free_gpus(12000, visible="3,5", rows=ROWS) == ["3", "5"]
    assert free_gpus(12000, visible="5,3", rows=ROWS) == ["5", "3"]


def test_a_full_gpu_inside_the_allocation_is_skipped():
    assert free_gpus(12000, visible="2,3", rows=ROWS) == ["3"]


def test_an_empty_allocation_selects_nothing():
    assert free_gpus(12000, visible="", rows=ROWS) == []
