color_balance
=============

Balance your colors!!

color_balance implements algorithms for reading and applying the color information between overlapping georeferenced images with support for masking out pixels.

## Development Environment

The development environment is provided as a [Vagrant](https://www.vagrantup.com/) VM. This VM installs all dependencies as specified in 'init.sh'.

To start the Vagrant VM, navigate to the root directory and type:
```
    vagrant up
```
Then ssh into the vagrant environment by typing:
```
    vagrant ssh
```
Finally, navigate to the root directory, which is shared between the host machine and VM:
```
    cd /vagrant
```

Nosetests can be used to run the unit tests:
```
    nosetests ./tests
```

## Packaging

The color_balance python package is hosted on [Pypi](https://pypi.python.org/pypi/color_balance/).

## Dependencies

python 2.7
python-numpy (>= 1.6)
python-gdal (>= 1.11)
python-opencv >=2.3.1-7
