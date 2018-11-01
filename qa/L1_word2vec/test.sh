#!/bin/bash

set -o pipefail
set -e

redir=(tee /dev/tty)
echo CHECKING FOR TTY | tee /dev/tty || redir=(cat)

python word2vec_basic.py | ${redir[@]} \
  | tail -n 16 | grep "Nearest to seven" \
  | awk '/nine/ && /eight/ && /six/ && /five/' \
  | grep -q .
