
To run the grid domain, use the below command, with optional command line arguments:
```
python -m dreamcoder.domains.grid.grid --task tree --enumerationTimeout 30 -i 2 -c 2
```

You can run tests for the python implementation too, requires pytest installed via `pip install pytest`
```
make test
```

### Command-line arguments
* `--task`, one of `tree`, `people_gibbs`, `grammar`
* `--grammar`. `pen` is the default grammar, and includes turn, move, embed, and pen up/down. `nopen` leaves out pen up/down. `pen_setloc` includes pen up/down as well as a setlocation primitive. When `pen_setloc` is active, the start board & location of the task set is wiped out.
* `--try_all_start`, executes program from all start locations and picks the best one to start at. Impacts log likelihood evaluation.

#### Generic Dreamcoder Arguments
* `-i`, number of iterations
* `-c`, number of CPUs
* `--enumerationTimeout`, number of seconds that program enumeration executes for.
* `--recognitionTimeout`, number of seconds that program recognition executes for.

-   --task tree --enumerationTimeout 10 -i 2 -c 2 --grammar pen
