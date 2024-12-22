==============
casmsocial
==============


.. image:: https://img.shields.io/pypi/v/casmsocial.svg
        :target: https://pypi.python.org/pypi/casmsocial

.. image:: https://img.shields.io/travis/clinejc/casmsocial.svg
        :target: https://travis-ci.com/clinejc/casmsocial

.. image:: https://readthedocs.org/projects/casmsocial/badge/?version=latest
        :target: https://casmsocial.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




casmsocial is a Python framework for implementing agent-based models that simulate the dynamics of a synthetic population


* Free software: MIT license
* Documentation: https://casmsocial.readthedocs.io.


Features
--------

* TODO

Requirements
------------
`casmsocial` requires Python 3.11+.

Installation
------------

After cloning the GitHub repository https://github.com/ccgsem/casmsocial:

Next:

```
export CC=mpicxx; export CXX=mpicxx
cd casmsocial
poetry env use 3.12
poetry shell
poetry build
poetry install
```

To build a Docker image for `casmsocial`:

```
docker build -t casmsocial . -f dockerfile-mitre
```

Quickstart
----------

To run:

```
mpirun -n 1 python -m casmsocial.casmsocial config/casmsocial.yaml
```

To run in a Docker container:

```
docker run \
        -v `pwd`/config:/usr/src/app/config \
        -v ~/Library/CloudStorage/Box-Box/Predicting_Population_Response/data:/usr/src/app/data \
        --rm -it --entrypoint bash casmsocial
root@d332db7567f5:/usr/src/app# ./env/bin/poetry shell
root@d332db7567f5:/usr/src/app# mpirun -n 1 python -m casmsocial.casmsocial config/casmsocial_wc.yaml
```

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
