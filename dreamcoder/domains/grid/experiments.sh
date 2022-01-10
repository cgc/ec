# Very verbose, basically everything that isn't a very noisy status message.
export OCAMLRUNPARAM=v=0x5BD

function exp() {
  command python -m dreamcoder.domains.grid.grid -c 1 -i 4 --enumerationTimeout 60 "$@"
}

for task in discon; do
  for ppw in 0 2 3 10; do
    for prim in pen penctx explicit_mark; do
      exp --task $task --no-recognition --try_all_start --partial_progress_weight $ppw --grammar $prim |& tee ppw${ppw}-${task}-${prim}-norec.log
    done
  done
done

for task in discon grammar people_gibbs; do
  for ppw in 1 2 3 10; do
    exp --task $task --no-recognition --try_all_start --partial_progress_weight $ppw |& tee ppw${ppw}-${task}-norec.log
  done
done

for task in discon grammar people_gibbs; do
  exp --task $task --no-recognition |& tee ${task}-norec.log
  exp --task $task --no-recognition --try_all_start |& tee tas-${task}-norec.log
  exp --task $task --recognition |& tee ${task}-rec.log
  exp --task $task --recognition --try_all_start |& tee tas-${task}-rec.log

  for prim in penctx explicit_mark; do
    exp --task $task --no-recognition --try_all_start --grammar $prim |& tee tas-${task}-${prim}-norec.log
  done
done
