import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomRotation
import torchvision.transforms.functional as TF


SPLIT_PREFIX = {
    'train': 'pos_train',
    'val': 'vai_train',
    'test': 'test',
}

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ProcessedDSMDataset(Dataset):
    guide_channels = 4

    def __init__(
            self,
            data_dir: str,
            split='train',
            crop_size=250,
            scaling=10,
            do_horizontal_flip=True,
            max_rotation_angle=0.,
            in_memory=False,
            crop_deterministic=False,
            **kwargs
    ):
        if split not in SPLIT_PREFIX:
            raise ValueError(f'Unsupported split {split}')
        if scaling <= 0:
            raise ValueError('scaling must be positive')

        self.root = self._resolve_root(data_dir)
        self.split = split
        self.prefix = SPLIT_PREFIX[split]
        self.scaling = int(scaling)
        self.crop_size = None if crop_size is None or int(crop_size) <= 0 else int(crop_size)
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = float(max_rotation_angle)
        self.crop_deterministic = crop_deterministic

        if self.crop_size is not None and self.crop_size % self.scaling != 0:
            raise ValueError(f'crop_size ({self.crop_size}) must be divisible by scaling ({self.scaling})')

        self.records = self._build_records()
        self.cache = [self._load_record(record) for record in self.records] if in_memory else None
        self.deterministic_map = self._build_deterministic_map() if crop_deterministic and self.crop_size else None

    @staticmethod
    def _resolve_root(data_dir):
        root = Path(data_dir)
        if (root / 'pos_train_DSM_HR').is_dir():
            return root
        nested = root / 'ProcessedData_scale10'
        if nested.is_dir():
            return nested
        raise FileNotFoundError(f'Cannot find ProcessedData_scale10 dataset under {root}')

    @staticmethod
    def _sample_id(path):
        matches = re.findall(r'\d+', path.stem)
        if not matches:
            raise ValueError(f'Cannot parse numeric sample id from {path.name}')
        return int(matches[-1])

    def _index_folder(self, suffix):
        folder = self.root / f'{self.prefix}_{suffix}'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing required folder: {folder}')
        return {self._sample_id(path): path for path in sorted(folder.glob('*.tif'))}

    def _build_records(self):
        dsm_hr = self._index_folder('DSM_HR')
        dsm_lr = self._index_folder('DSM_LR')
        rgb = self._index_folder('RGB')
        adapter = self._index_folder('adapter_guide')

        ids = sorted(set(dsm_hr) & set(dsm_lr) & set(rgb) & set(adapter))
        if not ids:
            raise RuntimeError(f'No complete DSM samples found for split {self.split} in {self.root}')

        sam3 = self._optional_index_folder('SAM3')
        label = self._optional_index_folder('label')
        records = []
        for sample_id in ids:
            record = {
                'id': sample_id,
                'dsm_hr': dsm_hr[sample_id],
                'dsm_lr': dsm_lr[sample_id],
                'rgb': rgb[sample_id],
                'adapter': adapter[sample_id],
            }
            if sample_id in sam3:
                record['sam3'] = sam3[sample_id]
            if sample_id in label:
                record['label'] = label[sample_id]
            records.append(record)
        return records

    def _optional_index_folder(self, suffix):
        folder = self.root / f'{self.prefix}_{suffix}'
        if not folder.is_dir():
            return {}
        return {self._sample_id(path): path for path in sorted(folder.glob('*.tif'))}

    def _build_deterministic_map(self):
        deterministic_map = []
        for record_index, record in enumerate(self.records):
            with Image.open(record['dsm_hr']) as image:
                width, height = image.size
            if self.crop_size > height or self.crop_size > width:
                raise ValueError(f'crop_size ({self.crop_size}) is larger than sample {record["id"]}: {height}x{width}')
            num_h = height // self.crop_size
            num_w = width // self.crop_size
            deterministic_map.extend(
                (record_index, crop_h, crop_w)
                for crop_h in range(num_h)
                for crop_w in range(num_w)
            )
        return deterministic_map

    def __getitem__(self, index):
        if self.deterministic_map is None:
            record_index = index
            crop_index = None
        else:
            record_index, crop_h, crop_w = self.deterministic_map[index]
            crop_index = (crop_h, crop_w)

        sample = self.cache[record_index] if self.cache is not None else self._load_record(self.records[record_index])
        sample = {key: value.clone() if torch.is_tensor(value) else value for key, value in sample.items()}
        sample = self._crop(sample, crop_index)
        sample = self._augment(sample)
        sample = self._finalize(sample)
        return sample

    def __len__(self):
        return len(self.deterministic_map) if self.deterministic_map is not None else len(self.records)

    def _load_record(self, record):
        rgb = self._read_rgb(record['rgb'])
        adapter = self._read_single(record['adapter'])
        guide = torch.cat([rgb, adapter], dim=0)

        sample = {
            'id': record['id'],
            'guide': guide,
            'y': self._read_single(record['dsm_hr']),
            'source': self._read_single(record['dsm_lr']),
        }
        if 'sam3' in record:
            sample['sam3'] = self._read_rgb(record['sam3'], normalize=False)
        if 'label' in record:
            sample['label'] = self._read_single(record['label']).long()
        return sample

    @staticmethod
    def _read_rgb(path, normalize=True):
        with Image.open(path) as image:
            array = np.array(image.convert('RGB'), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return (tensor - RGB_MEAN) / RGB_STD if normalize else tensor

    @staticmethod
    def _read_single(path):
        with Image.open(path) as image:
            array = np.array(image, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array).unsqueeze(0)

    def _crop(self, sample, crop_index):
        if self.crop_size is None:
            return sample

        _, hr_h, hr_w = sample['y'].shape
        _, lr_h, lr_w = sample['source'].shape
        lr_crop = self.crop_size // self.scaling

        if lr_crop > lr_h or lr_crop > lr_w:
            raise ValueError(f'LR crop ({lr_crop}) is larger than source sample: {lr_h}x{lr_w}')

        if crop_index is None:
            lr_top = torch.randint(0, lr_h - lr_crop + 1, (1,)).item()
            lr_left = torch.randint(0, lr_w - lr_crop + 1, (1,)).item()
        else:
            crop_h, crop_w = crop_index
            lr_top = crop_h * lr_crop
            lr_left = crop_w * lr_crop

        hr_top = lr_top * self.scaling
        hr_left = lr_left * self.scaling
        hr_bottom = min(hr_top + self.crop_size, hr_h)
        hr_right = min(hr_left + self.crop_size, hr_w)
        lr_bottom = lr_top + (hr_bottom - hr_top) // self.scaling
        lr_right = lr_left + (hr_right - hr_left) // self.scaling

        sample['guide'] = sample['guide'][:, hr_top:hr_bottom, hr_left:hr_right]
        sample['y'] = sample['y'][:, hr_top:hr_bottom, hr_left:hr_right]
        sample['source'] = sample['source'][:, lr_top:lr_bottom, lr_left:lr_right]
        if 'sam3' in sample:
            sample['sam3'] = sample['sam3'][:, hr_top:hr_bottom, hr_left:hr_right]
        if 'label' in sample:
            sample['label'] = sample['label'][:, hr_top:hr_bottom, hr_left:hr_right]
        return sample

    def _augment(self, sample):
        if self.do_horizontal_flip and self.split == 'train' and torch.rand(()) < 0.5:
            for key in ('guide', 'y', 'source', 'sam3', 'label'):
                if key in sample:
                    sample[key] = sample[key].flip(-1)

        if self.max_rotation_angle > 0 and self.split == 'train':
            angle = RandomRotation.get_params([-self.max_rotation_angle, self.max_rotation_angle])
            sample['guide'] = TF.rotate(sample['guide'], angle, InterpolationMode.BILINEAR, fill=0)
            sample['y'] = TF.rotate(sample['y'], angle, InterpolationMode.BILINEAR, fill=0)
            sample['source'] = TF.rotate(sample['source'], angle, InterpolationMode.BILINEAR, fill=0)
            if 'sam3' in sample:
                sample['sam3'] = TF.rotate(sample['sam3'], angle, InterpolationMode.BILINEAR, fill=0)
            if 'label' in sample:
                sample['label'] = TF.rotate(sample['label'].float(), angle, InterpolationMode.NEAREST, fill=0).long()
        return sample

    def _finalize(self, sample):
        y = sample['y']
        source = sample['source']

        mask_hr = torch.isfinite(y) & (y > 0)
        mask_lr = torch.isfinite(source) & (source > 0)
        y = torch.where(mask_hr, y, torch.zeros_like(y))
        source = torch.where(mask_lr, source, torch.zeros_like(source))

        y_bicubic = F.interpolate(
            source.unsqueeze(0),
            size=y.shape[-2:],
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        sample['y'] = y
        sample['source'] = source
        sample['mask_hr'] = mask_hr.float()
        sample['mask_lr'] = mask_lr.float()
        sample['y_bicubic'] = y_bicubic
        return sample
