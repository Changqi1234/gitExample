#!/bin/bash

out="hw4best100.csv"

ls *.csv | grep -v "$out" | head -n 1 | xargs head -n 1 > "$out"

grep -vh "^distance" $(ls *.csv | grep -v "$out") \
    | sort -t, -k1,1g \
    | head -n 100 >> "$out"
