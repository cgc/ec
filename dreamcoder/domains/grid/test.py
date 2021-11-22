import numpy as np
import grid

def test_basics():
    start = np.zeros((3, 3))
    location = (1, 0)
    goal = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    program = grid.parseGrid('((grid_move) (grid_right) (grid_move))')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location)).grid == goal)
    assert grid.GridTask("test case", start, goal, location).logLikelihood(program) == -3

    # Testing inverse temperature here too
    assert grid.GridTask("test case", start, goal, location, invtemp=5).logLikelihood(program) == -15

def test_pen():
    start = np.zeros((3, 3))
    location = (1, 1)
    goal = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('((grid_move) (grid_right) (grid_dopenup) (grid_move) (grid_right) (grid_dopendown) (grid_move))')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location)).grid == goal)
    # pen instructions accumulate cost
    assert grid.GridTask("test case", start, goal, location).logLikelihood(program) == -7

def test_grid_embed():
    start = np.zeros((3, 3))
    location = (2, 0)
    goal = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    program = grid.parseGrid('''(
        (
            grid_embed
            (grid_move)
            (
                grid_embed
                (grid_move)
            )
            (grid_right)
            (grid_move)
        )
        (grid_right)
        (grid_move)
    )''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location)).grid == goal)
    # Accumulates reward appropriately for grid_embed
    assert grid.GridTask("test case", start, goal, location).logLikelihood(program) == -6

def test_setlocation():
    start = np.zeros((3, 3))

    program = grid.parseGrid('''((grid_setlocation 0 1))''')
    assert grid.executeGrid(program, grid.GridState(start, (2, 0))) is None, 'Can only setlocation if it is not set'

    for primitive in ['grid_move', 'grid_dopenup', 'grid_dopendown', 'grid_right', 'grid_left']:
        assert grid.executeGrid(
            grid.parseGrid(f'(({primitive}))'),
            grid.GridState(start, (-1, -1))) is None, f'No {primitive} before a setlocation'

    goal1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('''((grid_setlocation 1 2))''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, (-1, -1))).grid == goal1)

    goal2 = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('''((grid_setlocation 1 2) (grid_move))''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, (-1, -1))).grid == goal2)

def test_try_all_start():
    start = np.zeros((3, 3))
    goal = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    program = grid.parseGrid('((grid_move) (grid_right) (grid_move))')
    assert grid.GridTask("test case", start, goal, (-1, -1), try_all_start=True).logLikelihood(program) == -3

    # a bit silly
    assert grid.GridTask("test case", start, start, (-1, -1), try_all_start=True).logLikelihood(program) == -float('inf')
