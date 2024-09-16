import pytest

from liteinfer.kvcache.block import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager


def make_manager(num_blocks=8, block_size=4):
    return BlockSpaceManager(BlockAllocator(num_blocks, block_size))


def test_append_grows_block_table():
    mgr = make_manager(num_blocks=8, block_size=4)
    mgr.add_sequence(0)
    # first 4 tokens share one block, 5th spills to a second block
    tables = []
    for _ in range(5):
        block_id, slot = mgr.append_token(0)
        tables.append((block_id, slot))
    assert len(mgr.get(0).block_table) == 2
    assert tables[0][1] == 0 and tables[3][1] == 3
    assert tables[4][1] == 0  # wrapped into slot 0 of the new block


def test_slot_for_matches_append():
    mgr = make_manager()
    mgr.add_sequence(1)
    appended = [mgr.append_token(1) for _ in range(6)]
    for pos, expected in enumerate(appended):
        assert mgr.slot_for(1, pos) == expected


def test_free_returns_blocks_to_pool():
    alloc = BlockAllocator(4, 4)
    mgr = BlockSpaceManager(alloc)
    mgr.add_sequence(0)
    for _ in range(8):
        mgr.append_token(0)
    assert alloc.num_free == 2
    mgr.free_sequence(0)
    assert alloc.num_free == 4


def test_duplicate_sequence_raises():
    mgr = make_manager()
    mgr.add_sequence(0)
    with pytest.raises(ValueError):
        mgr.add_sequence(0)
