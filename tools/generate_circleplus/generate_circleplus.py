import argparse
import sys
from circle_plus_builder import CirclePlusBuilder
from train_info_builder import TrainInfoBuilder, TrainInfoLoss, TrainInfoOptimizer, TrainInfoLossReduction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate_circleplus', description='Generate circleplus file')
    parser.add_argument('input', help='Input circle file')
    parser.add_argument('output', help='Output circle+ file')

    # loss parser
    parser.add_argument(
        'loss',
        choices=TrainInfoLoss.types(),
        default=TrainInfoLoss.default(),
        help='Loss')

    # optimizer parser
    optimizer_subparsers = parser.add_subparsers(dest='optimizer', help='Optimizer')
    for opt in TrainInfoOptimizer.types():
        opt_parser = optimizer_subparsers.add_parser(opt)
        for option, value in TrainInfoOptimizer.arguments(opt).items():
            opt_parser.add_argument('--' + option, default=value)

    # batch size parser
    parser.add_argument('--batch_size', default=32, help='batch size')

    # loss reduction parser
    parser.add_argument(
        '--loss_reduction',
        choices=TrainInfoLossReduction.types(),
        default=TrainInfoLossReduction.default(),
        help='Loss reduction type')

    args = parser.parse_args()

    tbuilder = TrainInfoBuilder(
        lossfn=TrainInfoLoss.lossfn(args.loss),
        lossfnOptType=TrainInfoLoss.lossfnOptType(args.loss),
        lossfnOpt=TrainInfoLoss.lossfnOpt(args.loss, args),
        optimizer=TrainInfoOptimizer.optimizer(args.optimizer),
        optimizerOptType=TrainInfoOptimizer.optimizerOptType(args.optimizer),
        optimizerOpt=TrainInfoOptimizer.optimizerOpt(args.optimizer, args),
        batchSize=int(args.batch_size),
        lossReductionType=TrainInfoLossReduction.lossReductionType(args.loss_reduction))

    builder = CirclePlusBuilder(args.input)
    builder.injectMetaData('CIRCLE_TRAINING', tbuilder.get())
    builder.export(args.output)
