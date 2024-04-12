#!/usr/bin/bash
x="gfm"
omniperf profile -n ${x}_${SLURM_PROCID} -- "$@"

