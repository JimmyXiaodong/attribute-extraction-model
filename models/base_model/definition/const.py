from functools import wraps


def const(cls):
    @wraps(cls)
    def new_setattr(name):
        raise Exception('const : {} can not be changed'.format(name))

    cls.__setattr__ = new_setattr
    return cls


@const
class PoolerType(object):
    TYPE_TUPLE = ("cls", "cls_before_pooler", "avg", "avg_top2",
                  "avg_first_last")
    CLS = "cls"
    CLS_BEFORE_POOLER = "cls_before_pooler"
    AVG = "avg"
    AVG_TOP2 = "avg_top2"
    AVG_FIRST_LAST = "avg_first_last"


@const
class ModelType(object):
    TYPE_TUPLE = ("bert", "roberta", "nezha", "electra", "longformer")
    BERT = "bert"
    ROBERTA = "roberta"
    NEZHA = "nezha"
    ELECTRA = "electra"
    LONGFORMER = "longformer"


@const
class _Const(object):
    POOLER_TYPE = PoolerType()
    MODEL_TYPE = ModelType()


CONST = _Const()
