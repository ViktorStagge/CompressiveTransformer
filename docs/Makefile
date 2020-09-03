.PHONY: install docs html clean help

SPHINX_BUILD ?= sphinx-build
SPHINX_OPTIONS ?=
BUILD_DIR ?= build
SOURCE_DIR ?= source

help:
	@$(SPHINX_BUILD) -M help "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_OPTIONS) $(0)

install:
	echo "Installing required packages for building the documentation"
	@cd ..; \
		make install;
	@pip install -r requirements.txt
	@echo "Installation complete."

docs: html

html: clean
	@echo "Creating Documentation"
	@$(SPHINX_BUILD) -M $@ "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_OPTIONS) $(O)

clean:
	-@rm -r $(BUILD_DIR) 2>/dev/null

