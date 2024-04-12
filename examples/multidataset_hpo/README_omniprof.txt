## INSTRUCTIONS TO ACTIVATRE "omniperf" FOR PROFILING

## Load module
ml omniperf/1.0.10

## Create a file called "wrap_omniperf.bash" within the directory where you  run the code, and the files must contain the following:
#!/usr/bin/bash
x="gfm"
omniperf profile -n ${x}_${SLURM_PROCID} -- "$@"


## Include "wrap_omniperf.bash" in the "srun" command, as follows:
srun -n$((SLURM_JOB_NUM_NODES*8)) --gpus-per-task=1 --gpu-bind=closest wrap_omniperf.bash /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/envs/hydragnn/bin/python -u gfm.py --multi --ddstore --multi_model_list="ANI1x"
