rm -rf dist
rm -rf build
python setup.py sdist
python setup.py bdist_wheel
twine check dist/*
twine upload dist/*