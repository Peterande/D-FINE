"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data


class DetDataset(data.Dataset):
    # Maximum number of images to cache per worker process.
    # With 4 workers, total cache is ~4x this. At ~1MB per decoded image,
    # 5000 items ≈ 5GB per worker ≈ 20GB total.
    _MAX_CACHE_ITEMS = 5000

    def __getitem__(self, index):
        # Cache decoded images in memory after first load to avoid repeated
        # disk I/O and JPEG decoding. Effective for finetuning datasets that
        # fit in RAM (500-10K images typical).
        cache = getattr(self, '_item_cache', None)
        if cache is None:
            self._item_cache = {}
            cache = self._item_cache

        if index in cache:
            import copy
            img, target = copy.deepcopy(cache[index])
        else:
            img, target = self.load_item(index)
            if len(cache) < self._MAX_CACHE_ITEMS:
                import copy
                cache[index] = copy.deepcopy((img, target))

        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError(
            "Please implement this function to return item before `transforms`."
        )

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1
