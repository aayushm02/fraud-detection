.PHONY: install train evaluate serve test clean all

install:
	pip install -r requirements.txt

train:
	python scripts/train.py --dataset both

evaluate:
	python scripts/evaluate.py --report

serve:
	python scripts/serve.py --dashboard

test:
	pytest tests/ -v

clean:
	rm -rf models/ reports/ __pycache__ .pytest_cache

all: install train evaluate test
