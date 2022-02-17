# Very verbose, basically everything that isn't a very noisy status message.
# export OCAMLRUNPARAM=v=0x5BD

SEED=334419183

# https://stackoverflow.com/questions/3004811/how-do-you-run-multiple-programs-in-parallel-from-a-bash-script
trap 'kill 0' SIGINT

function max_jobs {
   while [ `jobs -r -p | wc -l` -ge $1 ]; do
      sleep 1
   done
}

function exp() {
  # HACK: we randomly generate a log file name, to make sure they aren't overwritten
  # by other concurrent processes.
  LOGFILE=logoutput/$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32).log
  touch $LOGFILE
  command singularity-venv/bin/python3 -m dreamcoder.domains.grid.grid -c 1 -i 10 --enumerationTimeout 120 "$@" --log_file_path_for_mlflow $LOGFILE |& tee $LOGFILE
}

for recogflag in --no-recognition --recognition; do
  for batch in "" "--taskReranker randomShuffle --taskBatchSize 100 --seed $SEED"; do
    for arity in 1; do
      for task in discon_no_curr people_gibbs_discon people_gibbs_discon_500 people_gibbs_500; do
        for prim in pen explicit_mark; do
          for ppw in 0 10; do
            max_jobs 4
            sleep 1 # adding this so jobs are ordered in output
            exp $recogflag --task $task --try_all_start --partial_progress_weight $ppw --grammar $prim --arity $arity $batch &
          done
        done
      done
    done
  done
done

wait
