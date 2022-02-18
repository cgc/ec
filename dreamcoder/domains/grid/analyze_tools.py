from grid import *
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TaskFrontier:
    task: ...
    entries: ...

    def best_program_entry(self):
        return max(self.entries, key=lambda e: e.logPrior + e.logLikelihood)

    def is_solved(self):
        e = self.best_program_entry()
        return (e.logPrior + e.logLikelihood) > GridTask.incorrect_penalty

    def execute_best(self, **kw):
        e = self.best_program_entry()
        score, x, y = max(self.task._score_for_all_locations(e.program))
        s = execute_grid(e.program, self.task.start, (x, y), **kw)
        assert np.all(s.grid == self.task.goal) or score < GridTask.incorrect_penalty, (score, s, e.program)
        return dict(
            state=s,
            location=(x, y),
            score=score,
        )

def iter_frontiers(tasks, ecResult, *, iteration=-1):
    for task in tasks:
        es = ecResult.frontiersOverTime[task][iteration].entries
        if not es:
            continue
        tf = TaskFrontier(task, es)
        yield task, tf

class TracingInvented(Program):
    def __init__(self, stack, invented):
        self.stack = stack
        self.invented = invented
    def evaluate(self, environment):
        '''
        This wrapping implementation only works for extremely simple cases of reuse
        that involve tail calls to the continuation. Cases where the continuation is
        called mid-function will not be appropriately tagged.
        TODO: analyze programs to ensure they only have explicit tail calls to continuation?
        '''
        def wrapped(k):
            returned = False
            def wrappedk(arg):
                nonlocal returned
                assert not returned, 'Assume this is only called once'
                returned = True
                v = self.stack.pop()
                assert v == self.invented, 'Assume this is called in a straightforward way'
                return k(arg)

            fn = self.invented.evaluate(environment)(wrappedk)

            def wrapped_state_mapper(s):
                self.stack.append(self.invented)
                rv = fn(s)
                assert returned, 'Should have been called by now'
                return rv

            return wrapped_state_mapper
        return wrapped
    def show(self, isFunction): return f"Trace({self.invented.show(False)})"
    def inferType(self, *a, **k): return self.invented.inferType(*a, **k)

class StackTracingRewrite(object):
    def __init__(self, stack, continuationType=CONTINUATION_TYPE):
        self.stack = stack
        self.continuationType = continuationType
    def invented(self, e):
        assert e.tp == self.continuationType, 'Only analyzing simple inventions for now'
        return TracingInvented(self.stack, Invented(e.body.visit(self)))
    def primitive(self, e): return e
    def index(self, e): return e
    def application(self, e): return Application(e.f.visit(self), e.x.visit(self))
    def abstraction(self, e): return Abstraction(e.body.visit(self))

def execute_grid(p, start, location, *, trace=False):
    cls = GridState
    if trace:
        # This is a complete hack that's just meant to disambiguate
        # stack entries that would otherwise be identical (repeated calls to a routine)
        class ListWithCounter(list):
            def __init__(self):
                super().__init__()
                self.i = 0
            def append(self, x):
                self.i += 1
                return super().append((self.i, x))
            def pop(self):
                i, x = super().pop()
                return x
        stack = ListWithCounter()
        p = p.visit(StackTracingRewrite(stack))
        class TracingGridState(GridState):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if self.history is not None:
                    self.history[-1]['stack'] = list(stack)
        cls = TracingGridState

    return executeGrid(p, cls(start, (-1, -1), history=[]).setlocation(location))

def generate_grid_rect(grid, facecolor="none", edgecolor="k", linewidth=0.5):
    rv = []
    for s in np.ndindex(grid.shape):
        patch = plt.Rectangle((s[0]-0.5, s[1]-0.5), 1, 1, linewidth=linewidth, facecolor=facecolor, edgecolor=edgecolor)
        # ?? making sure the reactangle is within the bounds: https://stackoverflow.com/a/60577729 patch.set_clip_path(patch)
        rv.append((s, patch))
    return rv

def plot(start, end=None, history=None, *, size=3, ax=None):
    if isinstance(start, TaskFrontier):
        tf = start
        start, end, history = tf.task.start, tf.task.goal, tf.execute_best(trace=True)['state'].history

    w, h = start.shape
    if ax is None:
        aspect_ratio = w/h
        _, ax = plt.subplots(figsize=(size*aspect_ratio, size))
    ax.set(
        xticks=[],
        yticks=[],
        xlim=[-1/2, w-1/2],
        ylim=[-1/2, h-1/2],
    )
    for xy, r in generate_grid_rect(start):
        if start[xy]:
            r.set_facecolor('blue')
        elif end[xy]:
            r.set_facecolor((.8, .8, .8))
        ax.add_artist(r)
    if history:
        prog_to_idx = {}
        cmap = plt.get_cmap('tab20')

        try:
            maxdepth = max(len(curr['stack']) for curr in history if 'stack' in curr)
        except ValueError:
            maxdepth = 0

        for previ, (prev, curr) in enumerate(zip(history[:-1], history[1:])):
            curri = previ + 1
            if prev['location'] == (-1, -1):
                ax.scatter(*curr['location'], marker='*', color='red')
                continue

            xs = [prev['location'][0], curr['location'][0]]
            ys = [prev['location'][1], curr['location'][1]]

            if 'stack' not in curr:
                ax.plot(xs, ys, c='k')

            prev_stack = prev['stack']
            curr_stack = curr['stack']
            mul = 4
            ax.plot(xs, ys, c='k', lw=mul*(maxdepth+1))
            for i, curr_invented in enumerate(curr['stack']):
                xs_, ys_ = xs, ys
                if curr_stack[:i+1] != prev_stack[:i+1]:
                    alpha = 0.25
                    xs_ = [xs[0] * (1-alpha) + xs[1] * alpha, xs[1]]
                    ys_ = [ys[0] * (1-alpha) + ys[1] * alpha, ys[1]]

                lw = maxdepth - i # this maps i=0 to maxdepth and i=maxdepth-1 to 1
                # give this program an index (which we map to a color)
                if curr_invented not in prog_to_idx:
                    prog_to_idx[curr_invented] = len(prog_to_idx)
                c = cmap.colors[prog_to_idx[curr_invented]]
                ax.plot(xs_, ys_, c=c, lw=mul*lw, zorder=i+3) # 3+ seems necessary to get above default z values

def plot_trace(start, history=None, *, animate=False, size=None):
    if isinstance(start, TaskFrontier):
        tf = start
        start, history = tf.task.start, tf.execute_best(trace=True)['state'].history

    def render_step(ax, i):
        h_so_far = history[:i+1]
        assert len(h_so_far) == i+1
        last_state = h_so_far[-1]
        plot(start, last_state['grid'], h_so_far, ax=ax)
        #m = ['^', '>', 'v', '<'] # actually is rotate clockwise 90deg from this
        m = ['<', '^', '>', 'v'][last_state['orientation']]
        ax.scatter(*last_state['location'], c='r', zorder=2, marker=m, s=400/start.shape[0])

    w, h = start.shape
    aspect_ratio = w/h

    if animate:
        size = size or 2
        _, ax = plt.subplots(figsize=(size*aspect_ratio, size))
        return simple_animation(len(history), render_step, interval=100, ax=ax)
    else:
        size = size or 1
        f, axes = plt.subplots(1, len(history), figsize=(size * aspect_ratio * len(history), size))
        for i, ax in enumerate(axes):
            render_step(ax, i)

def simple_animation(n, fn, *, ax=None, filename=None, interval=100):
    '''
    This is a generic routine to make simple animations; it entirely wipes out & re-renders at every time step.
    Parameters:
    - n - number of frames.
    - fn(ax, i) - rendering callback, must take axis and rendering iteration as arguments.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if ax is None: _, ax = plt.subplots()
    f = ax.figure

    fn(ax, 0) # Running this once to make size

    def update(t):
        for a in ax.lines + ax.collections:
            a.remove()
        fn(ax, t)
        return []

    a = FuncAnimation(
        f, update, frames=n, interval=interval, blit=True, repeat=False)
    plt.close()

    if filename is not None:
        assert filename.endswith('.gif') or filename.endswith('.mp4'), 'Only supports exporting to .gif or .mp4'
        if filename.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=1000./interval, bitrate=1800)
            a.save(filename, writer=writer)
            from IPython.display import Video
            return Video(filename)
        else:
            a.save(filename, writer='imagemagick')
            from IPython.display import Image
            return Image(filename)

    from IPython.display import HTML
    return HTML(a.to_html5_video())
