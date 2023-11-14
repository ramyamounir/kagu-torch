Dataset
#######


.. code-block:: python

    from kagu.datasets.kagu_dataset import KaguDatasetArguments, KaguDataset

    kagu_args = KaguDatasetArguments(dataset='data/kagu', 
                                      frame_size=[299,299], 
                                      world_size=4, 
                                      global_rank=0, 
                                      snippet=16, 
                                      step=8)
    kagu_dataset = kagu_dataset(kagu_args)

.. autoclass:: kagu.datasets.kagu_dataset.KaguDatasetArguments
    :members:


.. autoclass:: kagu.datasets.kagu_dataset.KaguDataset
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__


