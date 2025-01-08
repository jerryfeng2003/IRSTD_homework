import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import random

class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train', preprocessed=True):
        self.dataset_dir = args.dataset_dir
        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.preprocessed = preprocessed  
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        if mode == 'train':
            self.imgs_dir = os.path.join(self.dataset_dir, 'train', 'image')
            self.label_dir = os.path.join(self.dataset_dir, 'train', 'mask')
            self.names = sorted(os.listdir(self.imgs_dir))

            if not self.preprocessed:
                self.images, self.masks = self._load_and_preprocess_data(self.imgs_dir, self.label_dir)
                self._save_preprocessed_data(self.images, self.masks, 'train')
                # pass
            else:
                self.images, self.masks = self._load_preprocessed_data('train')

        elif mode == 'val':
            # self.test_dir = 'SIRST'
            self.test_dir = 'MDvsFA'
            self.names = []
            self.val_data = []  


            if not self.preprocessed:
                imgs_dir = os.path.join(self.dataset_dir, 'test', self.test_dir, 'image')
                label_dir = os.path.join(self.dataset_dir, 'test', self.test_dir, 'mask')
                self.names += sorted(os.listdir(imgs_dir))
                self.images, self.masks = self._load_and_preprocess_data(imgs_dir, label_dir)
                self.val_data.append((self.images, self.masks))
                self._save_preprocessed_data(self.images, self.masks, self.test_dir)
            else:
                self.images, self.masks=self._load_preprocessed_data(self.test_dir)

        else:
            raise ValueError("Unknown mode: Choose either 'train' or 'val'")



    def _load_and_preprocess_data(self, img_dir, mask_dir):
        """ Helper function to load and preprocess all images and masks """
        images = []
        masks = []
        for img_name in tqdm(sorted(os.listdir(img_dir)), desc="Processing images", ncols=100):
            img_path = osp.join(img_dir, img_name)
            if self.test_dir=='SIRST':
                mask_name=img_name.replace('.png', '_pixels0.png')
            else:
                mask_name=img_name
            mask_path = osp.join(mask_dir, mask_name)

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)

            if self.mode == 'train':
                img, mask = self._sync_transform(img, mask)
            elif self.mode == 'val':
                img, mask = self._testval_sync_transform(img, mask)

            img, mask = self.transform(img), transforms.ToTensor()(mask)

            images.append(img)
            masks.append(mask)

        return images, masks

    def _save_preprocessed_data(self, images, masks, mode):
        data_dir = os.path.join(self.dataset_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f'{mode}_preprocessed.pt')

        torch.save({
            'images': images,
            'masks': masks
        }, file_path)
        print(f"Preprocessed data saved to {file_path}")

    def _load_preprocessed_data(self, mode):
        data_dir = os.path.join(self.dataset_dir, 'data')
        file_path = os.path.join(data_dir, f'{mode}_preprocessed.pt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessed data not found: {file_path}")

        data = torch.load(file_path)
        return data['images'], data['masks']

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode=='val':
            return self.images[idx], self.masks[idx]

        else:
            raise ValueError("Unknown mode")

    def __len__(self):
        return len(self.images) 

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
