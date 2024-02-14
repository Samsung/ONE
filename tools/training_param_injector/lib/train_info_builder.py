import flatbuffers

from . import train_info as tinfo
from schema import circle_traininfo_generated as ctr


class Builder:
    TRAINING_FILE_IDENTIFIER = b"CTR1"

    def __init__(self, info: tinfo.TrainingInfo, buf_size=0):
        self.builder = flatbuffers.Builder(buf_size)
        train_info = self.__build_train_info(info)
        self.builder.Finish(train_info, self.TRAINING_FILE_IDENTIFIER)

    def __build_optimizer(self, optimizer: tinfo.Optimizer):

        if (isinstance(optimizer, tinfo.SGD)):
            ctr.SGDOptionsStart(self.builder)
            ctr.SGDOptionsAddLearningRate(self.builder, optimizer.learning_rate)
            sgd_option = ctr.SGDOptionsEnd(self.builder)
            return ctr.Optimizer.SGD, ctr.OptimizerOptions.SGDOptions, sgd_option

        elif (isinstance(optimizer, tinfo.Adam)):
            ctr.AdamOptionsStart(self.builder)
            ctr.AdamOptionsAddBeta1(self.builder, optimizer.beta1)
            ctr.AdamOptionsAddBeta2(self.builder, optimizer.beta2)
            ctr.AdamOptionsAddEpsilon(self.builder, optimizer.epsilon)
            ctr.AdamOptionsAddLearningRate(self.builder, optimizer.learning_rate)
            adam_option = ctr.AdamOptionsEnd(self.builder)
            return ctr.Optimizer.ADAM, ctr.OptimizerOptions.AdamOptions, adam_option

        else:
            raise ValueError(f"unknown optimizer: {type(optimizer)}")

    def __build_loss(self, loss: tinfo.Loss):
        if (isinstance(loss, tinfo.SparseCategoricalCrossentropy)):
            ctr.SparseCategoricalCrossentropyOptionsStart(self.builder)
            ctr.SparseCategoricalCrossentropyOptionsAddFromLogits(
                self.builder, loss.from_logits)
            sparse_cce = ctr.SparseCategoricalCrossentropyOptionsEnd(self.builder)
            return ctr.LossFn.SPARSE_CATEGORICAL_CROSSENTROPY, ctr.LossFnOptions.SparseCategoricalCrossentropyOptions, sparse_cce

        elif (isinstance(loss, tinfo.CategoricalCrossentropy)):
            ctr.CategoricalCrossentropyOptionsStart(self.builder)
            ctr.CategoricalCrossentropyOptionsAddFromLogits(self.builder,
                                                            loss.from_logits)
            cce = ctr.CategoricalCrossentropyOptionsEnd(self.builder)
            return ctr.LossFn.CATEGORICAL_CROSSENTROPY, ctr.LossFnOptions.CategoricalCrossentropyOptions, cce

        elif (isinstance(loss, tinfo.MeanSquaredError)):
            ctr.MeanSquaredErrorOptionsStart(self.builder)
            mse = ctr.MeanSquaredErrorOptionsEnd(self.builder)
            return ctr.LossFn.MEAN_SQUARED_ERROR, ctr.LossFnOptions.MeanSquaredErrorOptions, mse

        else:
            raise ValueError(f"unknown loss: {type(loss)}")

    def __build_loss_rdt(self, loss_rdt: tinfo.LossReduction):
        if (loss_rdt == tinfo.LossReduction.SUM_OVER_BATCH_SIZE):
            return ctr.LossReductionType.SumOverBatchSize
        elif (loss_rdt == tinfo.LossReduction.SUM):
            return ctr.LossReductionType.Sum
        else:
            raise ValueError(f"unkonw loss reduction: {loss_rdt}")

    def __build_train_info(self, info: tinfo.TrainingInfo):

        optimizer_args = self.__build_optimizer(info.optimizer)
        loss_args = self.__build_loss(info.loss)
        loss_rdt = self.__build_loss_rdt(info.loss.reduction)

        ctr.ModelTrainingStart(self.builder)

        # optimizer
        optimizer, optimizer_opt_t, optimizer_opt = optimizer_args
        ctr.ModelTrainingAddOptimizer(self.builder, optimizer)
        ctr.ModelTrainingAddOptimizerOptType(self.builder, optimizer_opt_t)
        ctr.ModelTrainingAddOptimizerOpt(self.builder, optimizer_opt)

        # loss
        loss, loss_opt_t, loss_opt = loss_args
        ctr.ModelTrainingAddLossfn(self.builder, loss)
        ctr.ModelTrainingAddLossfnOptType(self.builder, loss_opt_t)
        ctr.ModelTrainingAddLossfnOpt(self.builder, loss_opt)

        # to-be removed
        ctr.ModelTrainingAddEpochs(self.builder, 0)

        # others
        ctr.ModelTrainingAddBatchSize(self.builder, info.batch_size)
        ctr.ModelTrainingAddLossReductionType(self.builder, loss_rdt)

        model_training = ctr.ModelTrainingEnd(self.builder)
        return model_training

    def get_buff(self):
        return self.builder.Output()
