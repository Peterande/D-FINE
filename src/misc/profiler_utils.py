"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg, 
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    
    model_for_info = copy.deepcopy(cfg.model).deploy()
        
    flops, macs, _ = calculate_flops(model=model_for_info, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
