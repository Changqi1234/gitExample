#!/bin/bash

out="hw4best100.csv"

echo "\"distance\",\"spectrumID\",\"i\"" > "$out"

for f in *.csv; do
    if [[ "$f" != "$out" ]]; then
        tail -n +2 "$f"
    fi
done \
    | sort -t, - -k1,1g \
    | head -n 100 >> "$out"
