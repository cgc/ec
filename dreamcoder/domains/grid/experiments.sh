function exp() {
  command python -m dreamcoder.domains.grid.grid -c 1 -i 4 --enumerationTimeout 60 "$@"
}

exp --task grammar --no-recognition |& tee grammar-norec.log
exp --task people_gibbs --no-recognition |& tee people_gibbs-norec.log
exp --task grammar |& tee grammar-rec.log
exp --task people_gibbs |& tee people_gibbs-rec.log

exp --task grammar --no-recognition --try_all_start |& tee tas-grammar-norec.log
exp --task people_gibbs --no-recognition --try_all_start |& tee tas-people_gibbs-norec.log
exp --task grammar --try_all_start |& tee tas-grammar-rec.log
exp --task people_gibbs --try_all_start |& tee tas-people_gibbs-rec.log

exp --task grammar --no-recognition --try_all_start --partial_progress_weight 1 |& tee ppw1-grammar-norec.log
exp --task grammar --no-recognition --try_all_start --partial_progress_weight 2 |& tee ppw2-grammar-norec.log
exp --task grammar --no-recognition --try_all_start --partial_progress_weight 3 |& tee ppw3-grammar-norec.log
exp --task grammar --no-recognition --try_all_start --partial_progress_weight 10 |& tee ppw10-grammar-norec.log

exp --task people_gibbs --no-recognition --try_all_start --partial_progress_weight 1 |& tee ppw1-people_gibbs-norec.log
exp --task people_gibbs --no-recognition --try_all_start --partial_progress_weight 2 |& tee ppw2-people_gibbs-norec.log
exp --task people_gibbs --no-recognition --try_all_start --partial_progress_weight 3 |& tee ppw3-people_gibbs-norec.log
exp --task people_gibbs --no-recognition --try_all_start --partial_progress_weight 10 |& tee ppw10-people_gibbs-norec.log
