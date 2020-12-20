import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def sort_files(s):
    split = s.rsplit('_')
    file_num = int(split[-3])
    frame_num = int(split[-1])
    return file_num*200 + frame_num

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.file_path_list.sort(key=sort_files)

    def __getitem__(self, index):
        return torch.load(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class WaymoDataLoader():
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def collect_fn(self, batch):
        voxels_list = []
        voxel_features_list = []
        point_cloud_list = []
        foreground_list = []
        gt_reg_list = []
        gt_class_list = []
        gt_dir_list = []
        gt_label_list = []
        for i, d in enumerate(batch):
            v = d['voxels']
            voxels_list.append(torch.cat(((i*torch.ones(v.size(0), 1)).int(), v), dim=1))
            voxel_features_list.append(d['features'])
            point_cloud_list.append(d['point_cloud'])
            foreground_list.append(d['foreground'])
            gt_reg_list.append(d['gt_reg'])
            gt_class_list.append(d['gt_class'])
            gt_dir_list.append(d['gt_dir'])
            gt_label_list.append(d['gt_label'])


        data = {"voxel_coords": torch.cat(voxels_list, dim=0),
                "voxel_features": torch.cat(voxel_features_list, dim=0),
                "point_cloud": point_cloud_list,
                "foreground": foreground_list,
                "gt_reg": torch.stack(gt_reg_list),
                "gt_class": torch.stack(gt_class_list),
                "gt_dir": torch.stack(gt_dir_list),
                "gt_label": gt_label_list,
                "batch_size": len(batch)
                }

        return data

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn, 
                          pin_memory=True)

class WaymoDataModule(pl.LightningDataModule):
	def __init__(self, train_path, test_path, train_batch_size, test_batch_size=1, num_workers = 1, shuffle_train = True, shuffle_test = False):
		super().__init__()
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.num_workers = num_workers
		self.train_dataset = WaymoDataset(train_path)
		self.test_dataset = WaymoDataset(test_path)
		self.shuffle_train = shuffle_train
		self.shuffle_test = shuffle_test
		

	def train_dataloader(self):
		return WaymoDataLoader(self.train_dataset, self.train_batch_size, self.shuffle_train, self.num_workers).loader()

	def val_dataloader(self):
		return WaymoDataLoader(self.test_dataset, self.test_batch_size, self.shuffle_test, self.num_workers).loader()
