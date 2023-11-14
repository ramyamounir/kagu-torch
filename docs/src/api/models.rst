Models
######

Backbone Encoder
================


.. code-block:: python

    from kagu.models import BackboneArguments, Backbone

    backbone_args = BackboneArguments(backbone='inception',
                                   backbone_pretrained=True,
                                   backbone_frozen=True,
                                   )
    backbone = Backbone(args=backbone_args)

.. autoclass:: kagu.models.model.BackboneArguments
    :members:

.. autoclass:: kagu.models.model.Backbone
    :members:
    :special-members:
    :private-members:
    :exclude-members: __init__, __weakref__


====


Kagu Model
==========

.. code-block:: python

    from kagu.models import KaguModelArguments, KaguModel

    kagu_args = BackboneArguments(
                            predictor = 'lstm'
                            dropout= 0.4, 
                            teacher= True,
                            step= 8,
                                   )
    kagu_model = KaguModel(args=kagu_args)


.. autoclass:: kagu.models.model.KaguModelArguments
    :members:

.. autoclass:: kagu.models.model.KaguModel
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__


