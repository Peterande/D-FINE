import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None 
    return m.to(device) 


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg
        self.coco_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        self.obj365_ids = [
            0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96, 
            144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146, 
            204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2, 
            50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30, 
            169, 70, 328, 226
        ]
    def _setup(self):
        """Avoid instantiating unnecessary classes"""
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model

        # NOTE: Must load_tuning_state before EMA instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(
            self.model.to(device), sync_bn=cfg.sync_bn, find_unused_parameters=cfg.find_unused_parameters
        )

        self.criterion = self.to(cfg.criterion, device)
        self.postprocessor = self.to(cfg.postprocessor, device)

        self.ema = self.to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

    def cleanup(self):
        if self.writer:
            atexit.register(self.writer.close)

    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # NOTE: Instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, module, device):
        return module.to(device) if hasattr(module, 'to') else module

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def load_state_dict(self, state):
        """Load state dict, train/eval"""
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')

    def load_resume_state(self, path: str):
        """Load resume"""
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        self.load_state_dict(state)

    def load_tuning_state(self, path: str):
        """Load model for tuning and adjust mismatched head parameters"""
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist_utils.de_parallel(self.model)

        # Load the appropriate state dict
        if 'ema' in state:
            pretrain_state_dict = state['ema']['module']
        else:
            pretrain_state_dict = state['model']

        # Adjust head parameters between datasets
        adjusted_state_dict = self._adjust_head_parameters(module.state_dict(), pretrain_state_dict)
        stat, infos = self._matched_state(module.state_dict(), adjusted_state_dict)

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def _adjust_head_parameters(self, cur_state_dict, pretrain_state_dict):
        """Adjust head parameters between datasets."""
        # List of parameters to adjust
        if pretrain_state_dict['decoder.denoising_class_embed.weight'].size() != \
                cur_state_dict['decoder.denoising_class_embed.weight'].size():
            del pretrain_state_dict['decoder.denoising_class_embed.weight']

        head_param_names = [
            'decoder.enc_score_head.weight',
            'decoder.enc_score_head.bias'
        ]
        for i in range(8):
            head_param_names.append(f'decoder.dec_score_head.{i}.weight')
            head_param_names.append(f'decoder.dec_score_head.{i}.bias')

        adjusted_params = []

        for param_name in head_param_names:
            if param_name in cur_state_dict and param_name in pretrain_state_dict:
                cur_tensor = cur_state_dict[param_name]
                pretrain_tensor = pretrain_state_dict[param_name]
                adjusted_tensor = self.map_class_weights(cur_tensor, pretrain_tensor)
                if adjusted_tensor is not None:
                    pretrain_state_dict[param_name] = adjusted_tensor
                    adjusted_params.append(param_name)
                else:
                    print(f"Cannot adjust parameter '{param_name}' due to size mismatch.")

        return pretrain_state_dict

    def map_class_weights(self, cur_tensor, pretrain_tensor):
        """Map class weights from pretrain model to current model based on class IDs."""
        if pretrain_tensor.size() == cur_tensor.size():
            return pretrain_tensor

        adjusted_tensor = cur_tensor.clone()
        adjusted_tensor.requires_grad = False

        if pretrain_tensor.size() > cur_tensor.size():
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[coco_id] = pretrain_tensor[obj_id+1]
        else:
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[obj_id+1] = pretrain_tensor[coco_id]           

        return adjusted_tensor

    def fit(self):
        raise NotImplementedError('')

    def val(self):
        raise NotImplementedError('')