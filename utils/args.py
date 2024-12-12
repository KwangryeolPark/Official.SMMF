import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='smmf', type=str, help='Optimizer', choices=['adam', 'smmf'])
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning-rate for Adam and SMMF.')
    parser.add_argument('--beta1', default=0.9, type=float, help='First momentum coefficient for Adam and SMMF.')
    parser.add_argument('--beta2', default=0.999, type=float, help='Second momentum coefficient for Adam.')
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon regularization constant for Adam and SMMF.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight-decay for Adam and SMMF.')
    parser.add_argument('--decay_rate', default=-0.8, type=float, help='Decay-rate for SMMF.')
    parser.add_argument('--growth_rate', default=0.999, type=float, help='Growth-rate for SMMF.')
    parser.add_argument('--vector_reshape', default=True, type=bool, help='Vector reshape for SMMF.')
    
    parser.add_argument('--seed', default=0, type=int, help='Seed.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
    parser.add_argument('--num_epochs', default=200, type=int, help='The number of epochs.')
    
    args = parser.parse_args()
    return args
