"""by lyuwenyu
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.core import YAMLConfig

import torch
import torch.nn as nn 
import time
import tqdm
from calflops import calculate_flops


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=None)

    # NOTE load train mode state -> convert to deploy mode
    # cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    cfg2 = YAMLConfig(args.config, resume=None)
    class Model_for_flops(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg2.model.deploy()
            
        def forward(self, images):
            outputs = self.model(images)
            return outputs  

    model = Model().cuda()

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    data = torch.rand(1, 3, 640, 640).cuda()

    model2 = Model_for_flops().eval().cuda()

    # FPS test
    @torch.no_grad()
    def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=10):
        for iter_ in tqdm.tqdm(range(warm_iters + num_iters)):
            if iter_ == warm_iters:
                torch.cuda.synchronize()
                t_ = time.perf_counter()
            model(inputs, torch.tensor([[640, 640]]).cuda())

        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        return t / num_iters
    
    flops, macs, _ = calculate_flops(  model=model2, 
                                            input_shape=(1, 3, 640, 640),
                                            output_as_string=True,
                                            output_precision=4)
    params = sum(p.numel() for p in model2.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    
    t = measure_average_inference_time(model, data, 1000000)
    print('Average inference time: ', t)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='/home/pengys/code/rtdetrv2_pytorch/configs/dfine/dfine_hgnetv2_b4_6x_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    args = parser.parse_args()

    main(args)