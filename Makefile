PROJECT_NAME=CompressiveTransformer

USER=vs
REMOTE_SERVER=adrian_minimal
REMOTE_FOLDER=CompressiveTransformer


.PHONY: install train evaluate notebook remote-sync sync-remote

install:
	echo "Not Implemented: Installing CompressiveTransformer dependencies"

train:
	python train.py --kwarg1 something

evaluate:
	@echo "Not implemented"

notebook:
	jupyter notebook --port=7888

remote-sync:
	@rsync -avze ssh \
		--exclude 'data/*' \
		--exclude '*.ipynb' \
		--exclude '.git/' \
		--exclude '.idea/' \
		--exclude '*.pyc' \
		--include '*.py' \
		--progress \
		. \
		$(USER)@$(REMOTE_SERVER):~/CompressiveTransformer

sync-remote: remote-sync