pip install langchain langchain-openai langchain-community bentoml Image

python setup.py sdist bdist_wheel
pip install twine
twine upload dist/*