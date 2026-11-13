"""Bounded M4 CPU workflows using synthetic inputs and temporary weights."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from hyso import ModelBuilder
from hyso.models import MBConv, FusedMBConv, DropBlock2D, reparameterize_model


SCENARIOS = (
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_v2',
    'ghostnet', 'shufflenet_v2', 'repvgg', 'mobileone',
    'mbconv', 'dropblock', 'builder', 'state_dict',
)


def small_model(family: str, **options) -> torch.nn.Module:
    """Build a one-channel, three-class CPU demonstration configuration."""
    params = dict(input_channels=1, num_classes=3)
    if family == 'shufflenet_v2':
        params['variant'] = 'x0_5'
    else:
        params['width_mult'] = .125
    if family in ('efficientnet_v2', 'repvgg', 'mobileone'):
        params['depth_mult'] = .1
    params.update(options)
    return ModelBuilder.named(family, **params).build()


def main() -> None:
    """Run a selected short example with deterministic CPU tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=SCENARIOS, nargs='?', default='builder')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(13)
        x = torch.randn(2, 1, 32, 32)
        if args.scenario == 'mbconv':
            model = torch.nn.Sequential(MBConv(1, 8, expand_ratio=2), FusedMBConv(8, 8))
        elif args.scenario == 'dropblock':
            model = DropBlock2D(p=.2, block_size=3).train()
            output = model(x)
            assert torch.isfinite(output).all()
            assert torch.equal(model.eval()(x), x)
            print('dropblock', tuple(output.shape), 'finite; eval identity')
            return
        elif args.scenario == 'builder':
            model = (ModelBuilder((1, 32, 32), 3)
                     .add('mbconv', in_channels=1, out_channels=8, expand_ratio=2)
                     .add('ghost_conv', in_channels=8, out_channels=8)
                     .add('dropblock2d', p=.1, block_size=3).build())
        elif args.scenario.startswith('mobilenet_v3_'):
            model = small_model('mobilenet_v3', variant=args.scenario.rsplit('_', 1)[1])
        elif args.scenario == 'state_dict':
            model = small_model('mobilenet_v2').eval()
            with tempfile.TemporaryDirectory(prefix='hyso-m4-example-') as directory:
                config_path = Path(directory) / 'architecture.json'
                weights_path = Path(directory) / 'weights.pth'
                config_path.write_text(json.dumps(model.to_config()), encoding='utf-8')
                torch.save(model.state_dict(), weights_path)
                clone = ModelBuilder.from_json(config_path).build().eval()
                clone.load_state_dict(torch.load(weights_path, weights_only=True), strict=True)
                with torch.inference_mode():
                    assert torch.equal(model(x), clone(x))
            print('state_dict strict roundtrip OK; architecture and weights saved separately')
            return
        else:
            model = small_model(args.scenario)
        model.eval()
        with torch.inference_mode():
            output = model(x)
            if args.scenario in ('repvgg', 'mobileone'):
                deployed = reparameterize_model(model)
                converted = deployed(x)
                torch.testing.assert_close(converted, output, atol=2e-6, rtol=2e-5)
                clone = ModelBuilder.from_config(deployed.to_config()).build().eval()
                clone.load_state_dict(deployed.state_dict(), strict=True)
                assert torch.equal(clone(x), converted)
                print(args.scenario, tuple(converted.shape),
                      'deploy max_abs', (converted-output).abs().max().item())
            else:
                print(args.scenario, tuple(output.shape))


if __name__ == '__main__':
    main()
