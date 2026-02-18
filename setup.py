"""
CognitionOS - Setup Configuration
Provides backward compatibility and installation entry point
"""
from setuptools import setup, find_packages

# For backward compatibility and editable installs
setup(
    packages=find_packages(
        include=['core*', 'infrastructure*', 'services*', 'shared*'],
        exclude=['tests*', 'docs*', 'examples*']
    ),
)
