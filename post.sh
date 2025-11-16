#!/bin/bash
export LC_ALL=C
sort -m sorted_*.txt|uniq -c |sort -nr > countsOfWords 
#rm -rf chunk_*
#rm -rf sorted_*
#rm -rf *~
#rm -rf all_plays.txt
#rm -rf shakespeare*
#rm logs
