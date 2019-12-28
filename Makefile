install-deps:
	source venv/bin/activate; \
	pip install lock-requirements==0.1.1; \
	lock requirements.txt; \
	pip install -r requirements.txt; \

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

test:
	source venv/bin/activate; \
	py.test --verbose --color=yes ./tests; \

.PHONY: clean test install-deps