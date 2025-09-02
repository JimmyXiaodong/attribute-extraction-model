from functools import wraps
from models.base_model.definition.const import PoolerType, ModelType


def const(cls):
    @wraps(cls)
    def new_setattr(name):
        raise Exception('const : {} can not be changed'.format(name))

    cls.__setattr__ = new_setattr
    return cls


@const
class _ModelConst(object):
    IGNORED_KEYS = ["position_ids"]
    NEGATIVE_INPUT_SIZE = 3


@const
class _Const(object):
    MODEL_CONST = _ModelConst()
    POOLER_TYPE = PoolerType()
    MODEL_TYPE = ModelType()


CONST = _Const()
