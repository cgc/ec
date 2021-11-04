python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen > pen-recog.log 2>&1
python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen_setloc > penloc-recog.log 2>&1
python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen --no-recognition > pen-norecog.log 2>&1
python -m dreamcoder.domains.grid.grid --task grammar -c 10 --grammar pen_setloc --no-recognition > penloc-norecog.log 2>&1
