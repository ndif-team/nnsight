build:
	python -m build

publish:
	python -m twine upload dist/*
	__token__
	