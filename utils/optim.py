import torch
from smmf import SMMF

def get_optim(args, model):
    """Get optimizer

    Args:
        optim (str): adam or smmf
        args (Namespace): argument
    """
    optim = args.optimizer
    optim = optim.lower()
    if optim == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif optim == 'smmf':
        return SMMF(
            model.parameters(),
            lr=args.lr,
            beta=args.beta1,
            eps=args.eps,
            weight_decay=args.weight_decay,
            decay_rate=args.decay_rate,
            growth_rate=args.growth_rate,
            vector_reshape=args.vector_reshape
        )
    else:
        raise ValueError(f"{optim} is not implemented optimizer.")
    
