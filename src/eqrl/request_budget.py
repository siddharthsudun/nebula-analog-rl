"""Per-request cooperative search limit; a process watchdog bounds native calls."""
from contextvars import ContextVar
import time
active = ContextVar("eqrl_request_budget", default=None)

class SearchBudgetExpired(Exception):
    pass

class Budget:
    def __init__(self, deadline, reserve):
        self.deadline=deadline
        self.search_deadline=deadline-reserve
        self.records=[]
        self.attempts=0
    def bind(self, evaluate):
        def wrapped(*a, **kw):
            if time.monotonic() >= self.search_deadline:
                raise SearchBudgetExpired()
            self.attempts += 1
            result=evaluate(*a, **kw)
            if result[0] is not None:
                self.records.append(result[0])
            return result
        return wrapped
