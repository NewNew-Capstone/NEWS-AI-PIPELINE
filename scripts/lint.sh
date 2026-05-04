#!/bin/bash
set -e
ruff check .
python -m mypy .
