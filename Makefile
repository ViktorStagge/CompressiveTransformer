PROJECT_NAME=CompressiveTransformer

USER=vs
REMOTE_SERVER=adrian_minimal
REMOTE_FOLDER=CompressiveTransformer


.PHONY: train evaluate remote-sync sync-remote

train:
	python train.py --kwarg1 something

evaluate:
	@echo "Not implemented"

remote-sync:
	@rsync -avze ssh \
		--exclude 'data/*' \
		--exclude '*.ipynb' \
		--exclude '.git/' \
		--exclude '.idea/' \
		--include '*.py' \
		--progress \
		. \
		$(USER)@$(REMOTE_SERVER):~/CompressiveTransformer

sync-remote: remote-sync