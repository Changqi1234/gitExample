#!/bin/bash
export LC_ALL=C
sort -m sorted_*.txt|uniq -c |sort -nr > countsOfWords 
