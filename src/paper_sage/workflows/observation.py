"""将诊断日志绑定到当前节点执行上下文，不放入图状态或模型输入。"""

from contextvars import ContextVar
from functools import wraps


_active = ContextVar("paper_sage_observation", default=None)


def observe_operation(step, operation):
    active = _active.get()
    return operation() if active is None else active[0].call(step, operation)


def make_node_observer(log):
    def wrap(name, node):
        @wraps(node)
        def observed(state):
            token = _active.set((log, name))
            try:
                return log.call(name, lambda: node(state))
            finally:
                _active.reset(token)
        return observed
    return wrap


class ModelWithEvents:
    """仅代理本项目使用的绑定与 invoke；未处于已观察节点时直接调用原模型。"""

    def __init__(self, model):
        self.model = model

    def bind(self, *args, **kwargs):
        return ModelWithEvents(self.model.bind(*args, **kwargs))

    def bind_tools(self, *args, **kwargs):
        return ModelWithEvents(self.model.bind_tools(*args, **kwargs))

    def invoke(self, *args, **kwargs):
        active = _active.get()
        if active is None:
            return self.model.invoke(*args, **kwargs)
        log, node = active
        return log.call(f"model_{node}", lambda: self.model.invoke(*args, **kwargs))


def observe_model(model):
    return model if isinstance(model, ModelWithEvents) else ModelWithEvents(model)
