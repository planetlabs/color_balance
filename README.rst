=============
Color Balance
=============

color_balance implements algorithms for reading and applying the color information between overlapping georeferenced images with support for masking out pixels.


.. image:: https://pl-amit.s3.amazonaws.com/demo/color-balance/example.png


Installation
------------

.. code-block:: bash

    $ pip install color_balance


Example
-------

There is an example image set in ``tests/fixtures``.

.. code-block:: bash

    $ color-balance tests/fixtures/source.tif tests/fixtures/reference.tif tests/fixtures/adjusted.tif


Development
-----------

Tests may be run with [pytest](http://pytest.org/latest/).

.. code-block:: bash

    $ py.test tests
    
To install test dependencies:

.. code-block:: bash

    $ pip install -e .[test]