#!/bin/bash

tgzfile="$1"
name="$(basename "$tgzfile" .tgz)"

tar -xzf "$tgzfile"

template="/staging/groups/STAT_DSCP/boss/cB58_Lyman_break.fit"
data_dir="$name"

Rscript hw4.R "$template" "$data_dir"
