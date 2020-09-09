PROJECT_NAME=CompressiveTransformer

USER=vs
REMOTE_SERVER=adrian_minimal
REMOTE_FOLDER=CompressiveTransformer


.PHONY: install train evaluate notebook remote-sync sync-remote

install:
	echo "Not Implemented: Installing CompressiveTransformer dependencies"
	@pip install -r requirements.txt

train:
	python ct.py train

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
		--exclude 'docs/build/*' \
		--include '*.py' \
		--progress \
		. \
		$(USER)@$(REMOTE_SERVER):~/CompressiveTransformer

sync-remote: remote-sync