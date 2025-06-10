import argparse
from onert.experimental.train import session, DataLoader, optimizer, losses, metrics


def initParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--nnpkg',
                        required=True,
                        help='Path to the nnpackage file or directory')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to the file containing input data (e.g., .npy or raw)')
    parser.add_argument(
        '-l',
        '--label',
        required=True,
        help='Path to the file containing label data (e.g., .npy or raw).')
    parser.add_argument('--data_length', required=True, type=int, help='data length')
    parser.add_argument('--backends', default='train', help='Backends to use')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='epoch number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--loss', default='mse', choices=['mse', 'cce'])
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--loss_reduction_type', default='mean', choices=['mean', 'sum'])
    parser.add_argument('--validation_split',
                        default=0.0,
                        type=float,
                        help='validation split rate')

    return parser.parse_args()


def createOptimizer(optimizer_type, learning_rate=0.001, **kwargs):
    """
    Create an optimizer based on the specified type.
    Args:
        optimizer_type (str): The type of optimizer ('SGD' or 'Adam').
        learning_rate (float): The learning rate for the optimizer.
        **kwargs: Additional parameters for the optimizer.
    Returns:
        Optimizer: The created optimizer instance.
    """
    if optimizer_type.lower() == "sgd":
        return optimizer.SGD(learning_rate=learning_rate, **kwargs)
    elif optimizer_type.lower() == "adam":
        return optimizer.Adam(learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def createLoss(loss_type, reduction="mean"):
    """
    Create a loss function based on the specified type and reduction.
    Args:
        loss_type (str): The type of loss function ('mse', 'cce').
        reduction (str): Reduction type ('mean', 'sum').
    Returns:
        object: An instance of the specified loss function.
    """
    if loss_type.lower() == "mse":
        return losses.MeanSquaredError(reduction=reduction)
    elif loss_type.lower() == "cce":
        return losses.CategoricalCrossentropy(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train(args):
    """
    Main function to train the model.
    """
    # Create session and load nnpackage
    sess = session(args.nnpkg, args.backends)

    # Load data
    input_shape = sess.input_tensorinfo(0).dims
    label_shape = sess.output_tensorinfo(0).dims

    input_shape[0] = args.data_length
    label_shape[0] = args.data_length

    data_loader = DataLoader(args.input,
                             args.label,
                             args.batch_size,
                             input_shape=input_shape,
                             expected_shape=label_shape)
    print('Load data')

    # optimizer
    opt_fn = createOptimizer(args.optimizer, args.learning_rate)

    # loss
    loss_fn = createLoss(args.loss, reduction=args.loss_reduction_type)

    sess.compile(optimizer=opt_fn,
                 loss=loss_fn,
                 batch_size=args.batch_size,
                 metrics=[metrics.CategoricalAccuracy()])

    # Train model
    total_time = sess.train(data_loader,
                            epochs=args.epoch,
                            validation_split=args.validation_split,
                            checkpoint_path="checkpoint.ckpt")

    # Print timing summary
    print("=" * 35)
    print(f"MODEL_LOAD   takes {total_time['MODEL_LOAD']:.4f} ms")
    print(f"COMPILE      takes {total_time['COMPILE']:.4f} ms")
    print(f"EXECUTE      takes {total_time['EXECUTE']:.4f} ms")
    epoch_times = total_time['EPOCH_TIMES']
    for i, epoch_time in enumerate(epoch_times):
        print(f"- Epoch {i + 1}      takes {epoch_time:.4f} ms")
    print("=" * 35)

    print(f"nnpackage {args.nnpkg.split('/')[-1]} trains successfully.")


if __name__ == "__main__":
    args = initParse()

    train(args)
