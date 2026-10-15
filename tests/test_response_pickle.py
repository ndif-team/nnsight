"""``ResponseModel.pickle`` carries a ``bytes`` result at its own size.

At ``torch.save``'s default pickle protocol 2 a ``bytes`` value is written
through a latin-1 ``str``, about 1.5x its size.
"""

import os

from nnsight.schema.response import ResponseModel, Status


def test_bytes_data_round_trips_at_its_own_size():
    blob = os.urandom(1 << 20)
    response = ResponseModel(id="req-1", status=Status.COMPLETED, data=blob)

    wire = response.pickle()
    back = ResponseModel.unpickle(wire)

    assert back.id == "req-1"
    assert back.status == Status.COMPLETED
    assert back.data == blob
    assert len(wire) < len(blob) + 4096
