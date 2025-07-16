"""
Batch Generator.

Author: Heng Wang
Date: 3/4/2024
"""
import itertools
from typing import Optional


def batch(
    reader, 
    batch_size, 
    drop_last: Optional[bool] = False
):
    """

    This operator creates a batched reader.

    which combines the data from the input reader to batched data.

    Args:
        reader(generator): 
            the data reader to read from.
        batch_size(int): 
            size of each mini-batch.
        drop_last(bool, optional): 
            If set to True, the last batch is dropped when
            the size of last batch is not equal to batch_size, if set to False,
            it will not. Default: False.

    Returns:
        batch reader (generator)

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            def reader():
                for i in range(10):
                    yield i
            batch_reader = fluid.io.batch(reader, batch_size=2)

            for data in batch_reader():
                print(data)

            # Output is
            # [0, 1]
            # [2, 3]
            # [4, 5]
            # [6, 7]
            # [8, 9]
    """

    def batch_reader():  # noqa:WPS430
        b = []
        for instance in reader:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []

        if drop_last is False and b:
            cycle_it = itertools.cycle(iter(b))
            yield [next(cycle_it) for _ in range(batch_size)]

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(
            ''.join([
                'batch_size should be a positive integeral value, ',
                'but got batch_size={0}'.format(batch_size)
            ])
        )

    return batch_reader
