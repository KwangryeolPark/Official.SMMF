from utils.args import get_args
from utils.trainer import Trainer

def main():
    args = get_args()
    trainer = Trainer(args)
    
    trainer.fit()
    trainer.finish()
    
if __name__ == '__main__':
    main()
