
python word2vec_basic.py | tee /dev/tty \
  | tail -n 16 | grep "Nearest to seven" \
  | awk '/nine/ && /eight/ && /six/ && /five/' \
  | grep -q .
