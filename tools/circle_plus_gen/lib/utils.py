from typing import Union, Type, List
from schema import circle_traininfo_generated as ctr_gen

# yapf:disable
# type alias
OPTIM_OPTIONS_T = Union[Type[ctr_gen.SGDOptionsT],
                      Type[ctr_gen.AdamOptionsT]]

LOSSFN_OPTIONS_T = Union[Type[ctr_gen.SparseCategoricalCrossentropyOptionsT],
                         Type[ctr_gen.CategoricalCrossentropyOptionsT],
                         Type[ctr_gen.MeanSquaredErrorOptionsT]]
# yapf:enable


class OptimizerNamer:
    '''Return name(string) based on ModelTraining.OptimizerOpt'''
    names = {
        ctr_gen.SGDOptionsT: ['sgd', 'stocasticgradientdescent'],
        ctr_gen.AdamOptionsT: ['adam']
    }

    def __call__(self, opt: OPTIM_OPTIONS_T) -> List[str]:
        try:
            name = self.names[opt]
        except:
            print(f"unknown optimizer {type(opt)}")
        return name


class LossNamer:
    '''Return name(string) based on ModelTraining.LossfnOpt'''

    # yapf:disable
    names = {
        ctr_gen.SparseCategoricalCrossentropyOptionsT:
            ['sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'],
        ctr_gen.CategoricalCrossentropyOptionsT:
            ['categorical crossentropy', 'categoricalcrossentropy', 'cce'],
        ctr_gen.MeanSquaredErrorOptionsT:
            ['mean squared error', 'mse']
    }
    # yapf:eanble

    def __call__(self, lossfn:LOSSFN_OPTIONS_T) -> List[str]:
        try:
            name = self.names[lossfn]
        except:
            print(f"unknown lossfn {type(lossfn)}")
        return name


class LossReductionNamer:
    '''Return name(string) based on ModelTraining.LossReductionType '''
    names = {
        ctr_gen.LossReductionType.SumOverBatchSize: ['sum over batch size', 'sumoverbatchsize'],
        ctr_gen.LossReductionType.Sum: ['sum'],
    }

    def __call__(self, rdt:ctr_gen.LossReductionType) -> List[str]:
        try:
            name = self.names[rdt]
        except:
            print(f"unknown loss reduction type {rdt}")
        return name
