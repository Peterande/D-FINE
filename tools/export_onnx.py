"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        # cfg.model.load_state_dict(state, strict=False)
        model_dict = cfg.model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        cfg.model.load_state_dict(model_dict, strict=False)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    torch.onnx.export(
        model, 
        (data, size), 
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        dynamic = True 
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(args.output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='/home/pengys/code/rtdetrv2_pytorch/configs/dfine/dfine_hgnetv2_b4_6x_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', default="/home/pengys/code/rtdetrv2_pytorch/weight/b4/ema_0.9997_0.5394.pth", type=str, )
    parser.add_argument('--output_file', '-o', default='./b4-elan2x.onnx', type=str)
    parser.add_argument('--check',  action='store_true', default=True,)
    parser.add_argument('--simplify',  action='store_true', default=True,)

    args = parser.parse_args()
    # scp -i /home/pengys/code/rtdetrv2_pytorch/id_rsa -P 39097 /home/pengys/code/rtdetrv2_pytorch/b4_elanfl6.onnx pengys@122.51.61.156:/home/pengys/code/rtdetrv2/deployment
    main(args)
