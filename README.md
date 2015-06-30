# color_balance

Balance your colors!!

color_balance implements algorithms for reading and applying the color information between overlapping georeferenced images with support for masking out pixels.


## Installation

    pip install color_balance
    

## Development

Tests may be run with [pytest](http://pytest.org/latest/).

    py.test tests
    
To install test dependencies:

    pip install -e .[test]


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

## Dependencies

python 2.7
python-numpy (>= 1.6)
python-gdal (< 2.0)
