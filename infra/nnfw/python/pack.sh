#!/bin/bash

# packge
python3 setup.py sdist bdist_wheel

# deploy for TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# deploy for PyPI instead of TestPyPI
#twine upload dist/*
