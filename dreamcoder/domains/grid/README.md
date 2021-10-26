
To run the grid domain, use the below command, with optional command line arguments:
```
python -m dreamcoder.domains.grid.grid --task tree --enumerationTimeout 30 -i 2 -c 2
```

### Command-line arguments
* `--task`, one of `tree`, `people_gibbs`, `grammar`
* `--grammar`. `pen` is the default grammar, and includes turn, move, embed, and pen up/down. `nopen` leaves out pen up/down. `pen_setloc` includes pen up/down as well as a setlocation primitive. When `pen_setloc` is active, the start board & location of the task set is wiped out.

#### Generic Dreamcoder Arguments
* `-i`, number of iterations
* `-c`, number of CPUs
* `--enumerationTimeout`, number of seconds that program enumeratino executes for.

-   --task tree --enumerationTimeout 10 -i 2 -c 2 --grammar pen
