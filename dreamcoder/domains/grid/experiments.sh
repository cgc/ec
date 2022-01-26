# Very verbose, basically everything that isn't a very noisy status message.
# export OCAMLRUNPARAM=v=0x5BD

function exp() {
  command python -m dreamcoder.domains.grid.grid -c 1 -i 4 --enumerationTimeout 60 "$@" --log_file_path_for_mlflow output.log |& tee output.log
}

for task in discon_no_curr discon people_gibbs_discon people_gibbs; do
  for arity in 1 2 3; do
    for recogflag in --no-recognition --recognition; do
      for ppw in 0 1 2 3 10; do
        for prim in pen penctx explicit_mark; do
          exp $recogflag --task $task --try_all_start --partial_progress_weight $ppw --grammar $prim --arity $arity
        done
      done
    done
  done
done
