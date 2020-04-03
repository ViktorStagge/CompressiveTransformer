PROJECT_NAME=CompressiveTransformer

.PHONY: train evaluate

train:
	python train.py --kwarg1 something

evaluate:
	@echo "Not implemented"
