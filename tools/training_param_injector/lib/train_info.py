from enum import Enum
import copy
import json


# Utils
def get_int(msg: str, default: int) -> int:
    return int(input(f"{msg} ({str(default)}) : ") or default)


def get_float(msg: str, default: float) -> float:
    return float(input(f"{msg} ({str(default)}) : ") or default)


def get_bool(msg: str, default: bool) -> bool:
    res = default
    bool_input = input(f"{msg} ({str(default)}) : ")
    if bool_input.lower() in ['true', 'y', '1', 'Y', 'yes', 't']:
        res = True
    return res


# == Optimizer ==
class Optimizer(dict):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class SGD(Optimizer):
    name = ['sgd', 'stocasticgradientdescent']

    def __str__(self):
        return self.name[0]

    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    @staticmethod
    def create_from_input():
        return SGD(learning_rate=get_float("\t ㄴlearning_rate", 0.01))


class Adam(Optimizer):
    name = ['adam']

    def __str__(self):
        return self.name[0]

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @staticmethod
    def create_from_input():
        return Adam(
            learning_rate=get_float("\t ㄴlearning_rate", 0.001),
            beta1=get_float("\t ㄴbeta1", 0.9),
            beta2=get_float("\t ㄴbeta2", 0.999),
            epsilon=get_float("\t ㄴepsilon", 1e-07))


# == Loss ==
class LossReduction(Enum):
    SUM_OVER_BATCH_SIZE = 1
    SUM = 2

    def __str__(self):
        return str(self.name)

    @staticmethod
    def create_from_input():
        _input = input("\tㄴreduction(SUM_OVER_BATCH_SIZE, SUM) : ") or '1'
        if (_input in ['SUM_OVER_BATCH_SIZE', '1']):
            return LossReduction.SUM_OVER_BATCH_SIZE
        return LossReduction.SUM


class Loss:
    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE):
        self.reduction = reduction


class SparseCategoricalCrossentropy(Loss):
    name = [
        'sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'
    ]

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE, from_logits=False):
        super().__init__(reduction)
        self.from_logits = from_logits

    @staticmethod
    def create_from_input():
        return SparseCategoricalCrossentropy(
            reduction=LossReduction.create_from_input(),
            from_logits=get_bool("\tㄴfrom_logits", False))


class CategoricalCrossentropy(Loss):
    name = ['categorical crossentropy', 'categoricalcrossentropy', 'cce']

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE, from_logits=False):
        super().__init__(reduction)
        self.from_logits = from_logits

    @staticmethod
    def create_from_input():
        return CategoricalCrossentropy(
            reduction=LossReduction.create_from_input(),
            from_logits=get_bool("\tㄴfrom_logits", False))


class MeanSquaredError(Loss):
    name = ['mse', 'meansquarederror']

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE):
        super().__init__(reduction)

    @staticmethod
    def create_from_input():
        return MeanSquaredError(reduction=LossReduction.create_from_input())


def create_optimizer_interactive():
    while 1:
        keyword = input(f"optimizer (adam, sgd) : ") or "adam"
        if keyword.lower() in ['sgd']:
            return SGD.create_from_input()
        elif keyword.lower() in ['adam']:
            return Adam.create_from_input()
        else:
            print(f"unknown optimizer {keyword}")


def create_loss_interative():
    while 1:
        keyword = input(f"loss (mse, cce) : ") or "mse"
        if keyword.lower() in ['meansquarederror', 'mse']:
            return MeanSquaredError.create_from_input()
        elif keyword.lower() in ['cce']:
            return CategoricalCrossentropy.create_from_input()
        else:
            print(f'unknown loss {keyword}')


# == Training Information ==
class TrainingInfo:
    def __init__(self, optimizer: Optimizer, loss: Loss, batch_size: int):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size

    @staticmethod
    def create_from_input():
        print()
        print(f"Please input training parmameter you want to inject.")
        print(f"If you enter without entering, first parmater is always selected.")
        print()

        opt = create_optimizer_interactive()
        loss = create_loss_interative()
        batch_size = get_int("batch_size", 32)
        return TrainingInfo(opt, loss, batch_size)


# Training Information json format encoder
class JEcoder(json.JSONEncoder):
    def ecnode_opt(self, opt):
        ret = {}
        ret["type"] = str(opt)
        ret["args"] = copy.deepcopy(vars(opt))
        return ret

    def __encode_loss(self, loss):
        ret = {}
        ret["type"] = str(loss)
        ret["args"] = copy.deepcopy(vars(loss))
        ret["args"]["reduction"] = str(ret["args"]["reduction"])
        return ret

    def default(self, obj: TrainingInfo):
        ret = {}
        ret["optimizer"] = self.ecnode_opt(obj.optimizer)
        ret["loss"] = self.__encode_loss(obj.loss)
        ret["batch_size"] = obj.batch_size
        return ret
