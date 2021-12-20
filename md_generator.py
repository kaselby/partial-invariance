from meta_dataset.reader import Reader
from meta_dataset.dataset_spec import Split
from meta_dataset import dataset_spec as dataset_spec_lib

import torch
import os

DATASET_ROOT = "/ssd003/projects/meta-dataset"
ALL_DATASETS=["aircraft", "cu_birds", "dtd", "fungi", "ilsvrc_2012", "mscoco", "omniglot", "quickdraw", "traffic_sign", "vgg_flower"]


class MetaDatasetGenerator():
    N = len(ALL_DATASETS)
    def __init__(self, p_aligned=0.5, p_sameset=0.5, dataset_path=DATASET_ROOT, split=Split.TRAIN, device=torch.device('cpu')):
        self.split=split
        self.p_aligned = p_aligned
        self.p_sameset = p_sameset
        self.device=device
        self.dataset_specs = [dataset_spec_lib.load_dataset_spec(os.path.join(DATASET_ROOT, dataset)) for dataset in ALL_DATASETS]
        self.readers = [Reader(dataset_spec, split, False, 0) for dataset_spec in self.dataset_specs]
        self.datasets_by_class = [reader.construct_class_datasets() for reader in self.readers]

    def _generate(self, batch_size, set_size=(10,15)):
        def sample_dataset(dataset, data_class, n_samples):
            data_iter = iter(self.datasets_by_class[dataset][data_class])
            return [next(data_iter) for _ in range(n_samples)]

        aligned = (torch.rand(batch_size) < self.p_aligned)
        n_samples = torch.randint(*set_size).item()
        X = []
        Y = []
        for j in range(batch_size):
            if aligned[j]:
                dataset1 = torch.randint(self.N).item()
                dataset2 = dataset1
                class1 = torch.randint(len(self.datasets_by_class[dataset1])).item()
                class2 = class1
            else:
                if torch.rand() < self.p_sameset:
                    dataset1 = torch.randint(self.N).item()
                    dataset2 = dataset1
                    class1, class2 = torch.multinomial(torch.ones(len(self.datasets_by_class[dataset1])), 2)
                    class1, class2 = class1.item(), class2.item()
                else:
                    dataset1, dataset2 = torch.multinomial(torch.ones(self.N), 2)
                    dataset1, dataset2 = dataset1.item(), dataset2.item()
                    class1 = torch.randint(len(self.datasets_by_class[dataset1])).item()
                    class2 = torch.randint(len(self.datasets_by_class[dataset2])).item()
            X_j = sample_dataset(dataset1, class1, n_samples)
            Y_j = sample_dataset(dataset2, class2, n_samples)
            X.append(torch.stack(X_j, 0))
            Y.append(torch.stack(Y_j, 0))
        X = torch.stack(X, 0)
        Y = torch.stack(Y, 0)
        return (X.to(self.device),Y.to(self.device)), aligned.to(self.device)

    def __call__(self, *args, **kwargs):
        return self._generate(*args, **kwargs)
