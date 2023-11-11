build:
	rm -rf ./dist/
	python -m build

publish:
	python -m twine upload dist/*
	