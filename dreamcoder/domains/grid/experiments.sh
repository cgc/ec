#python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen --no-recognition |& tee pen-norecog.log
#python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen_setloc --no-recognition |& tee penloc-norecog.log
python -m dreamcoder.domains.grid.grid --task grammar -c 1 --grammar pen |& tee pen-recog.log
python -m dreamcoder.domains.grid.grid --task grammar -c 1 --grammar pen_setloc |& tee penloc-recog.log
