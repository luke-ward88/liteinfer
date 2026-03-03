import pytest

from liteinfer.errors import CacheFullError
from liteinfer.kvcache.blocks import BlockAllocator


def test_allocate_reduces_free_count():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.num_free == 4
    b = alloc.allocate()
    assert alloc.num_free == 3
    alloc.free(b)
    assert alloc.num_free == 4


def test_allocate_until_full_raises():
    alloc = BlockAllocator(num_blocks=2, block_size=8)
    alloc.allocate()
    alloc.allocate()
    with pytest.raises(CacheFullError):
        alloc.allocate()


def test_free_many_restores_pool():
    alloc = BlockAllocator(num_blocks=3, block_size=8)
    ids = [alloc.allocate() for _ in range(3)]
    alloc.free_many(ids)
    assert alloc.num_free == 3


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=0, block_size=8)
