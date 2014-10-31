try:
    from setuptools import setup
except:
    from distutils.core import setup


def parse_requirements(requirements_filename='requirements.txt'):
    requirements = []
    with open(requirements_filename) as requirements_file:
        for requirement in requirements_file:
            requirements.append(requirement.rstrip('\n'))
    return requirements


config = dict(
    name='color_balance',
    version=0.1.0
    url='https://github.com/planetlabs/color_balance',
    description='Color balancing',
    author='Jennifer Reiber Kyle',
    author_email='jennifer.kyle@planet.com',
    install_requires=parse_requirements(),
    packages=['color_balance'],
)

setup(**config)
