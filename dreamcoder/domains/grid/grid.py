from dreamcoder.task import *
from dreamcoder.program import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import *
import pickle, os
import joblib
from dreamcoder.utilities import numberOfCPUs

import torch.nn as nn
import torch.nn.functional as F
from dreamcoder.recognition import variable

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

class GridCNN(nn.Module):
    def __init__(self,tasks,testingTasks=[],cuda=False):
        super(GridCNN,self).__init__()
        self.CUDA=cuda
        self.recomputeTasks=True

        self.conv1=nn.Conv2d(1,16,3,stride=1)
        self.fc1=nn.Linear(64,16)



        #self.outputDimensionality=64
        self.outputDimensionality=16
        if cuda:
            self.CUDA=True
            self.cuda()

    def forward(self,v):
        assert v.shape[0]==4
        if len(v.shape)==2:
            v=np.expand_dims(v,(0,1))
            inserted_batch=True
        elif len(v.shape)==3:
            v=np.expand_dims(v,0)
            inserted_batch=True

        v=variable(v,cuda=self.CUDA).float()
        v=F.relu(self.conv1(v))
        v=torch.reshape(v,(-1,64))
        v=F.relu(self.fc1(v))

        if inserted_batch:
            return v.view(-1)
        else:
            return v
    def featuresOfTask(self, t):  # Take a task and returns [features]
        assert t.goal.shape[0]==4
        return self(t.goal)
    def featuresOfTasks(self, ts):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        assert ts[0].goal.shape[0]==4

        return self(np.array([t.goal for t in ts]))
    def taskOfProgram(self,p,t):

        start_state=GridState(np.zeros((4,4)),(1,1))
        p1=executeGrid(p,start_state)
        if p1 is None:
            print(f'program did not execute: {p}')
            return None
        t=GridTask("grid dream",start=start_state,goal=p1.grid,location=(1, 1))
        return t





class GridException(Exception):
    pass

currdir = os.path.abspath(os.path.dirname(__file__))

def tasks_from_grammar_boards():
    with open(f'{currdir}/grammar_boards.pkl', 'rb') as f:
        boards = pickle.load(f)

    for idx, (board, steps) in enumerate(boards.items()):
         board = np.asarray(board).reshape((4, 4))
         start = steps[0]
         loc = next(zip(*np.where(start)))
         yield GridTask(f'grammar_boards.pkl[{idx}]', start=start, goal=board, location=loc)

def tasks_people_gibbs():
    import numpy as np
    boards = np.load(f'{currdir}/people_sampled_boards.npy')
    for idx, board in enumerate(boards):
        start = np.zeros(boards.shape[1:])
        location = list(zip(*np.where(board)))[0] # arbitrarily pick a start spot
        start[location] = 1
        yield GridTask(
            f'people_sampled_boards.npy[{idx}]',
            start=start, goal=board, location=location)

def tree_tasks():
    st = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    loc = (2, 0)
    st[loc] = 1
    return [
        GridTask(f'left', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])),
        GridTask(f'right', start=st, location=loc, goal=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])),
        GridTask(f'both', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])),
        GridTask(f'both-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]])),
        GridTask(f'both-rightboth', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]])),
        GridTask(f'both-rightboth-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])),
    ]


class GridState:
    def __init__(self, start, location, *, orientation=0, pendown=True, history=None, reward=0.):
        self.grid = start
        self.location = location
        self.orientation = orientation % 4 # [0, 1, 2, 3]
        self.pendown = pendown
        self.reward = reward

        # HACK ordering of next statements is intentional to avoid saving `history` into the history.
        if history is not None:
            state_copy = dict(self.__dict__)
            assert 'history' not in state_copy, 'History should not include the `history` key.'
            history += [state_copy]
        self.history = history
    def next_state(self, **kwargs):
        a = dict(self.__dict__, **kwargs)
        if a.pop('step_cost', False):
            a['reward'] -= 1
        return GridState(a.pop('grid'), a.pop('location'), **a)
    def left(self):
        self._ensure_location()
        return self.next_state(orientation=self.orientation - 1, step_cost=True)
    def right(self):
        self._ensure_location()
        return self.next_state(orientation=self.orientation + 1, step_cost=True)
    def move(self):
        self._ensure_location()
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
        grid = self.grid
        if self.pendown:
            grid = np.copy(grid)
            grid[x, y] = 1
        return self.next_state(grid=grid, location=(x, y), step_cost=True)
    def dopendown(self):
        self._ensure_location()
        return self.next_state(pendown=True, step_cost=True)
    def dopenup(self):
        self._ensure_location()
        return self.next_state(pendown=False, step_cost=True)
    def setlocation(self, location):
        if self.location != (-1, -1):
            raise GridException('Location can only be set when unspecified.')
        grid = self.grid
        if self.pendown:
            grid = np.copy(grid)
            grid[location] = 1
        return self.next_state(grid=grid, location=location)

    def _ensure_location(self):
        if self.location == (-1, -1):
            raise GridException('Location must be set')
    def __repr__(self):
        return f'GridState({self.grid}, {self.location}, orientation={self.orientation}, pendown={self.pendown})'


class GridTask(Task):
    def __init__(self, name, start, goal, location, *, invtemp=1.):
        self.start = start
        self.goal = goal
        self.location = location
        self.invtemp = invtemp
        super().__init__(name, arrow(tgrid_cont,tgrid_cont), [], features=[])

    @property
    def specialTask(self):
        # Computing this dynamically since we modify the task when there's the option to set location.
        return ("GridTask", {
            "start": self.start.astype(np.bool).tolist(), "goal": self.goal.astype(np.bool).tolist(),
            "location": tuple(map(int, self.location)),
            "invtemp": self.invtemp,
        })

    def logLikelihood(self, e, timeout=None):
        yh = executeGrid(e, GridState(self.start, self.location), timeout=timeout)

        if yh is not None and np.all(yh.grid == self.goal): return self.invtemp * yh.reward
        return NEGATIVEINFINITY

def parseGrid(s):
    from sexpdata import loads, Symbol
    s = loads(s)
    _e = Program.parse("grid_embed")
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] in (
            Symbol("grid_right"), Symbol("grid_left"),
            Symbol("grid_move"), Symbol("grid_dopenup"),
            Symbol("grid_dopendown"),
        ):
            assert len(k) == 1
            return Application(Program.parse(k[0].value()),continuation)
        if k[0] in (
            Symbol('grid_setlocation'),
        ):
            assert len(k) == 3
            return Application(
                Application(
                    Application(Program.parse(k[0].value()), expression(k[1], environment)),
                    expression(k[2], environment)),
            continuation)
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



def _grid_left(k): return lambda s: k(s.left())
def _grid_right(k): return lambda s: k(s.right())
def _grid_move(k): return lambda s: k(s.move())
def _grid_dopendown(k): return lambda s: k(s.dopendown())
def _grid_dopenup(k): return lambda s: k(s.dopenup())
def _grid_setlocation(x): return lambda y: lambda k: lambda s: k(s.setlocation((x, y)))

def _grid_embed(body):
    def f(k):
        def g(s):
            s._ensure_location()

            identity = lambda x: x
            # TODO: use of identity here feels a bit heuristic; it's what tower's impl does, but it seems
            # to let misuse of the continuation happen in program induction (use of $0 and $1 in an embed
            # result in same value, but $1 should be incorrect & terminate program?)
            ns = body(identity)(s)
            # We keep the grid & reward state, but restore the agent state
            ns = s.next_state(grid=ns.grid, reward=ns.reward)
            return k(ns)
        return g
    return f

# TODO still not clear to me what types are doing in Python; how is this bound? Does it require definition in ocaml?
tgrid_cont = baseType("grid_cont")
primitives_base = [
    Primitive("grid_left", arrow(tgrid_cont, tgrid_cont), _grid_left),
    Primitive("grid_right", arrow(tgrid_cont, tgrid_cont), _grid_right),
    Primitive("grid_move", arrow(tgrid_cont, tgrid_cont), _grid_move),
    Primitive("grid_embed", arrow(arrow(tgrid_cont, tgrid_cont), tgrid_cont, tgrid_cont), _grid_embed),
]

primitives_pen = primitives_base + [
    Primitive("grid_dopendown", arrow(tgrid_cont, tgrid_cont), _grid_dopendown),
    Primitive("grid_dopenup", arrow(tgrid_cont, tgrid_cont), _grid_dopenup),
]

primitives_loc = primitives_pen + [
    Primitive("grid_setlocation", arrow(tint, tint, tgrid_cont, tgrid_cont), _grid_setlocation),
] + [
    Primitive(str(j), tint, j) for j in range(1,5) # HACK need to change this later?
]

def executeGrid(p, state, *, timeout=None):
    try:
        identity = lambda x: x
        return runWithTimeout(lambda : p.evaluate([])(identity)(state),
                              timeout=timeout)
    except RunWithTimeout: return None
    except GridException: return None

def parseArgs(parser):
    parser.add_argument(
        "-f",
        dest="DELETE_var",
        help="just adding this here to capture a jupyter notebook variable",
        default='x',
        type=str)
    parser.add_argument("--task", dest="task", default="grammar")
    parser.add_argument("--grammar", dest="grammar", default='pen', type=str)

if __name__ == '__main__':
    arguments = commandlineArguments(
        enumerationTimeout=120,
        solver='ocaml',
        compressor="ocaml",
        activation='tanh',
        iterations=5,
        recognitionTimeout=120,
        # TODO what does this arity do? seems to relate to grammar?
        a=3,
        maximumFrontier=5, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5,
        structurePenalty=1.,
        extras=parseArgs,
        featureExtractor=GridCNN,
        CPUs=numberOfCPUs(),
    )
    del arguments['DELETE_var']
    taskname = arguments.pop('task')

    grammar = arguments.pop('grammar')
    p = dict(
        no_pen=primitives_base,
        pen=primitives_pen,
        pen_setloc=primitives_loc,
    )[grammar]

    using_setloc = any(prim.name == 'grid_setlocation' for prim in p)

    # task dist
    train_dict = dict(
        grammar=tasks_from_grammar_boards(),
        people_gibbs=tasks_people_gibbs(),
        tree=tree_tasks(),
    )
    train = list(train_dict[taskname])
    if using_setloc:
        for task in train:
            task.start = np.zeros(task.start.shape)
            task.location = (-1, -1)
    test = train

    g0 = Grammar.uniform(
        p,
        # when doing grid_cont instead, we only consider $0
        # but when we only have type=tgrid_cont, then we get a nicer library for tree_tasks()
        continuationType=arrow(tgrid_cont,tgrid_cont))

    generator = ecIterator(g0, train,
                           testingTasks=test,
                           **arguments)
    for iter, result in enumerate(generator):
        print('-' * 100)
        print()
        print()
        print()
        print()
        dir = f'{currdir}/output'
        try:
            os.mkdir(dir)
        except:
            pass
        fn = f'{dir}/output-task{taskname}-iter{iter}-grammar{grammar}-recog{arguments["useRecognitionModel"]}.bin'
        joblib.dump(dict(result=result,train=train,arguments=arguments), fn)
