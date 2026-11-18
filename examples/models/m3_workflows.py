"""Run one bounded CPU model workflow; synthetic data illustrates API use only."""
import argparse
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso import ModelBuilder, TrainingConfig
from hyso.models import ResNet, DenseNet, BasicResidualBlock, SimAM


def main() -> None:
    """Demonstrate model selection, blocks, attention, Builder, training and weights."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario',choices=['resnet','variant','densenet','block','attention','builder','training','state_dict'],nargs='?',default='builder')
    parser.add_argument('--depth',type=int,choices=[18,34,50,101,152],default=18)
    args=parser.parse_args()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(12)
        x=torch.randn(2,1,32,32)
        if args.scenario in ('resnet','variant'):model=ResNet(depth=args.depth,input_channels=1,num_classes=3)
        elif args.scenario=='densenet':model=DenseNet(depth=121,input_channels=1,num_classes=3,growth_rate=4)
        elif args.scenario=='block':model=BasicResidualBlock(1,4,stride=2)
        elif args.scenario=='attention':model=torch.nn.Sequential(BasicResidualBlock(1,4),SimAM(4))
        else:model=ModelBuilder((1,32,32),3).add('basic_residual',in_channels=1,out_channels=4).add('simam',channels=4).build()
        if args.scenario=='training':
            loader=DataLoader(TensorDataset(x,torch.tensor([0,1])),batch_size=2)
            result=TrainingConfig(epochs=1,batch_size=2).prepare(model,loader).fit()
            print('synthetic CPU example updates:',result.to_dict()['applied_updates'])
        elif args.scenario=='state_dict':
            with tempfile.TemporaryDirectory(prefix='hyso-m3-example-') as directory:
                path=Path(directory)/'weights.pth';torch.save(model.state_dict(),path)
                clone=ModelBuilder.from_config(model.to_config()).build()
                clone.load_state_dict(torch.load(path,weights_only=True),strict=True)
                model.eval();clone.eval()
                with torch.inference_mode():assert torch.equal(model(x),clone(x))
            print('state_dict roundtrip OK; temporary example weights only')
        else:
            model.eval()
            with torch.inference_mode():print(args.scenario,tuple(model(x).shape))


if __name__=='__main__':main()
