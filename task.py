
EVALUATIONTABLE = {}

class RegressionTask():
    def __init__(self, name, request, examples):
        self.request = request
        self.name = name
        self.examples = examples
    def __str__(self): return self.name
    def check(self,e):
        f = e.evaluate([])
        for x,y in self.examples:
            if (x,e) in EVALUATIONTABLE: p = EVALUATIONTABLE[(x,e)]
            else:
                p = f(x)
                EVALUATIONTABLE[(x,e)] = p
            if p != y: return False
        return True
    def logLikelihood(self,e):
        if self.check(e): return 0.0
        else: return float('-inf')
