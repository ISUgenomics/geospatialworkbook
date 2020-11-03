#! /usr/bin/env bash

set -e
set -u

JUPYTER_NOTEBOOK=$1

jupyter nbconvert --to markdown ${JUPYTER_NOTEBOOK}
