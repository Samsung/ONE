import argparse
import re
import sys
import os
from circle_plus_builder import CirclePlusBuilder
from train_info_builder import TrainInfoBuilder, TrainInfoLoss, TrainInfoOptimizer, TrainInfoLossReduction


class StoreTrainableAction(argparse.Action):
    """Store trainable Operation index"""

    def __call__(self, parser, namespace, values, option_string=None):
        regex = re.compile(r'[^0-9-,]')
        if len(regex.findall(values)) != 0:
            sys.exit('{}: error: argument trainable: invalid format'.format(
                os.path.basename(__file__).split('.')[0]))
        strs = values.split(',')
        lists = []
        for s in strs:
            if '-' in s:
                start, end = s.split('-')
                for i in range(int(start), int(end) + 1):
                    lists.append(i)
            else:
                lists.append(int(s))
        setattr(namespace, self.dest, lists)


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

    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument(
        '--loss_reduction',
        choices=TrainInfoLossReduction.types(),
        default=TrainInfoLossReduction.default(),
        help='Loss reduction type')
    # TODO Support multiple trainable operation index
    parser.add_argument(
        '--trainable',
        action=StoreTrainableAction,
        help='Indexes of trainable nodes in graph\n'
        'The indexes can be passed as a comma-separated list or range form\n'
        'e.g. 1,2,3 or 1,2-5')

    args = parser.parse_args()

    tbuilder = TrainInfoBuilder(
        lossfn=TrainInfoLoss.lossfn(args.loss),
        lossfnOptType=TrainInfoLoss.lossfnOptType(args.loss),
        lossfnOpt=TrainInfoLoss.lossfnOpt(args.loss, args),
        optimizer=TrainInfoOptimizer.optimizer(args.optimizer),
        optimizerOptType=TrainInfoOptimizer.optimizerOptType(args.optimizer),
        optimizerOpt=TrainInfoOptimizer.optimizerOpt(args.optimizer, args),
        batchSize=int(args.batch_size),
        lossReductionType=TrainInfoLossReduction.lossReductionType(args.loss_reduction),
        trainable=args.trainable)

    builder = CirclePlusBuilder(args.input)
    builder.injectMetaData('CIRCLE_TRAINING', tbuilder.get())
    builder.export(args.output)
