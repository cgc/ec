from dreamcoder.task import *
from dreamcoder.program import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import *
import pickle, os

currdir = os.path.abspath(os.path.dirname(__file__))

def tasks_from_grammar_boards():
    with open(f'{currdir}/grammar_boards.pkl', 'rb') as f:
        boards = pickle.load(f)

    for idx, (board, steps) in enumerate(boards.items()):
         board = np.asarray(board).reshape((4, 4))
         start = steps[0]
         loc = next(zip(*np.where(start)))
         yield GridTask(f'grammar_boards.pkl[{idx}]', start=start, goal=board, location=loc)

def tree_tasks():
    st = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    loc = (2, 0)
    return [
#        GridTask(f'left', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])),
#        GridTask(f'right', start=st, location=loc, goal=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])),
        GridTask(f'both', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])),
        GridTask(f'both-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]])),
        GridTask(f'both-rightboth', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]])),
        GridTask(f'both-rightboth-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])),
    ]


class GridState:
    def __init__(self, start, location, *, orientation=0, pendown=True):
        self.grid = start
        self.location = location
        self.orientation = orientation # [0, 1, 2, 3]
        self.pendown = pendown
    def left(self):
        return GridState(self.grid, self.location, orientation=self.orientation - 1, pendown=self.pendown)
    def right(self):
        return GridState(self.grid, self.location, orientation=self.orientation + 1, pendown=self.pendown)
    def move(self):
        dx, dy = [
            (-1, 0), # up
            (0, +1), # right
            (+1, 0), # down
            (0, -1), # left
        ][self.orientation]
        xlim, ylim = self.grid.shape
        prevx, prevy = self.location
        x = prevx + dx
        if not (0 <= x < xlim):
            x = prevx
        y = prevy + dy
        if not (0 <= y < ylim):
            y = prevy
        if self.pendown:
            grid = np.copy(self.grid)
            grid[x, y] = 1
        return GridState(grid, (x, y), orientation=self.orientation, pendown=self.pendown)
    def dopendown(self):
        return GridState(self.grid, self.location, orientation=self.orientation, pendown=True)
    def dopenup(self):
        return GridState(self.grid, self.location, orientation=self.orientation, pendown=False)
    def __repr__(self):
        return f'GridState({self.grid}, {self.location}, orientation={self.orientation}, pendown={self.pendown})'


class GridTask(Task):
    def __init__(self, name, start, goal, location):
        self.start = start
        self.goal = goal
        self.location = location
        super().__init__(name, arrow(tline_cont,tline_cont), [], features=[])
        self.specialTask = ("GridTask", {"start": self.start, "goal": self.goal, "location": self.location})

    def logLikelihood(self, e, timeout=None):
        yh = executeGrid(e, GridState(self.start, self.location), timeout=timeout)
        if yh is not None and np.all(yh.grid == self.goal): return 0.
        return NEGATIVEINFINITY

def parseGrid(s):
    from sexpdata import loads, Symbol
    s = loads(s)
    _e = Program.parse("grid_embed")
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] in (Symbol("grid_right"), Symbol("grid_left"), Symbol("grid_move"), Symbol("grid_dopenup"), Symbol("grid_dopendown")):
            assert len(k) == 1
            return Application(Program.parse(k[0].value()),continuation)
        if k[0] == Symbol("grid_embed"):
            # TODO issues with incorrect continuations probably need to be dealt with here
            # I think the issue is that we hardcode Index(0)?
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_e,Abstraction(body)),continuation)
        assert False

    def expression(e, environment):
        for n, v in enumerate(environment):
            if e == v: return Index(n)
        if isinstance(e,int): return Program.parse(str(e))

        assert isinstance(e,list)
        if e[0] == Symbol('+'): return Application(Application(_addition, expression(e[1], environment)),
                                                   expression(e[2], environment))
        if e[0] == Symbol('-'): return Application(Application(_subtraction, expression(e[1], environment)),
                                                   expression(e[2], environment))
        assert False

    def block(b, environment, continuation):
        if len(b) == 0: return continuation
        return command(b[0], environment, block(b[1:], environment, continuation))

    return Abstraction(block(s, [], Index(0)))
    #try: return Abstraction(command(s, [], Index(0)))
    #except: return Abstraction(block(s, [], Index(0)))



def make_continuation(prop):
    return lambda k: lambda s: k(getattr(s, prop)())

def _embed(body):
    def f(k):
        def g(s):
            identity = lambda x: x
            # TODO: use of identity here feels a bit heuristic; it's what tower's impl does, but it seems
            # to let misuse of the continuation happen in program induction (use of $0 and $1 in an embed
            # result in same value, but $1 should be incorrect & terminate program?)
            ns = body(identity)(s)
            # We keep the grid state, but restore the agent state
            ns = GridState(ns.grid, s.location, orientation=s.orientation, pendown=s.pendown)
            return k(ns)
        return g
    return f

# TODO still not clear to me what types are doing in Python; how is this bound? Does it require definition in ocaml?
tline_cont = baseType("grid_cont")
primitives = [
    Primitive(f"grid_{prop}", arrow(tline_cont, tline_cont), make_continuation(prop))
    for prop in ['left', 'right', 'move', 'dopendown', 'dopenup']
] + [
    Primitive("grid_embed", arrow(arrow(tline_cont, tline_cont), tline_cont, tline_cont), _embed),
]

def executeGrid(p, state, *, timeout=None):
    try:
        identity = lambda x: x
        return runWithTimeout(lambda : p.evaluate([])(identity)(state),
                              timeout=timeout)
    except RunWithTimeout: return None
    except: return None

if __name__ == '__main__':
    # this is just making sure this is all wired up.
    start = np.zeros((2, 2))
    location = (1, 0)
    goal = np.copy(start)
    goal[0, :] = 1
    program = parseGrid('((grid_move) (grid_right) (grid_move))')
    assert np.all(executeGrid(program, GridState(start, location)).grid == goal)
    assert GridTask("test case", start, goal, location).logLikelihood(program) == 0

    start = np.zeros((3, 3))
    location = (2, 0)
    goal = np.copy(start)
    goal[0, 0] = goal[1, 0] = goal[1, 1] = goal[2, 1] = 1
    program = parseGrid('''(
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
    assert np.all(executeGrid(program, GridState(start, location)).grid == goal)

    # Done with tests above

    train = [
        GridTask("test case", start, goal, location)
    ]
    train = list(tasks_from_grammar_boards())
    train = tree_tasks()
    test = train

    g0 = Grammar.uniform(primitives, continuationType=tline_cont)
    arguments = commandlineArguments(
        #iterations=1,
        #enumerationTimeout=1,
        #maximumFrontier=10,
        enumerationTimeout=90, activation='tanh',
        iterations=3, recognitionTimeout=3600,
        # TODO what does this arity do? seems to relate to grammar?
        a=3,
        maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
               solver='python',
               compressor="pypy",
        CPUs=1)
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           **arguments)
    for result in generator:
        print('another iter')
        print('-' * 100)
        print('-' * 100)
        print()
        print()
