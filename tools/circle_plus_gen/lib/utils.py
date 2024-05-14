from schema import circle_traininfo_generated as ctr_gen


class OptimizerNamer:
    '''Return name(string) based on ModelTraining.OptimizerOpt'''
    names = {ctr_gen.SGDOptionsT: 'sgd', ctr_gen.AdamOptionsT: 'adam'}

    def __call__(cls, opt):
        try:
            name = cls.names[type(opt)]
        except:
            print(f"unknown optimizer {type(opt)}")
        return name


class LossNamer:
    '''Return name(string) based on ModelTraining.LossfnOpt'''
    names = {
        ctr_gen.SparseCategoricalCrossentropyOptionsT: 'sparse categorical crossentropy',
        ctr_gen.CategoricalCrossentropyOptionsT: 'categorical crossentorpy',
        ctr_gen.MeanSquaredErrorOptionsT: 'mean squared error'
    }

    def __call__(cls, lossfn):
        try:
            name = cls.names[type(lossfn)]
        except:
            print(f"unknown lossfn {type(lossfn)}")
        return name


class LossReductionNamer:
    '''Return name(string) based on ModelTraining.LossReductionType '''
    names = {
        ctr_gen.LossReductionType.SumOverBatchSize: 'SumOverBatchSize',
        ctr_gen.LossReductionType.Sum: 'Sum',
    }

    def __call__(cls, rdt):
        try:
            name = cls.names[rdt]
        except:
            print(f"unknown loss reduction type {rdt}")
        return name
