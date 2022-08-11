from meta_dataset.reader import Reader, parse_record
from meta_dataset.dataset_spec import Split
from meta_dataset import dataset_spec as dataset_spec_lib
from meta_dataset.transform import get_transforms

import torch
import os

DATASET_ROOT = "/ssd003/projects/meta-dataset"
ALL_DATASETS=["aircraft", "cu_birds", "dtd", "fungi", "ilsvrc_2012", "mscoco", "omniglot", "quickdraw", "traffic_sign", "vgg_flower"]

def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class MetaDatasetGenerator():
    def __init__(self, image_size=84, root_dir=DATASET_ROOT, split=Split.TRAIN):
        self.split=split
        self.image_size = image_size
        self.datasets_by_class = self._build_datasets(root_dir)
        self.N = len(self.datasets_by_class)
        self.transforms = get_transforms(self.image_size, self.split)

    def _build_datasets(self, root_dir, min_class_examples=20):
        datasets = []
        for dataset in ALL_DATASETS:
            dataset_path = os.path.join(root_dir, dataset)
            if os.path.exists(dataset_path):
                dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_path)
                reader = Reader(dataset_spec, self.split, False, 0) 
                class_datasets = reader.construct_class_datasets()
                if len(class_datasets) > 0:
                    split_classes = dataset_spec.get_classes(self.split)
                    filtered_class_datasets = [x for i, x in enumerate(class_datasets) if dataset_spec.get_total_images_per_class(split_classes[i]) >= min_class_examples]
                    if len(filtered_class_datasets) > 0:
                        datasets.append(filtered_class_datasets)
        return datasets
    
    def get_episode(self, n_classes, n_datasets):
        class_datasets=[]
        datasets = torch.multinomial(torch.ones(self.N), n_datasets) if n_datasets < self.N else torch.arange(self.N)
        n_datasets = len(datasets)
        
        N_remaining = n_classes
        for i in range(n_datasets):
            dataset_i = datasets[i].item()
            n_i = len(self.datasets_by_class[dataset_i])
            p_i = min(1., N_remaining/(n_datasets-i)/n_i)
            m_i = torch.distributions.Binomial(n_i, torch.tensor([p_i])).sample().int().item()
            if m_i > 0:
                classes_i = torch.multinomial(torch.ones(n_i), m_i)
                class_datasets.append([self.datasets_by_class[dataset_i][j.item()] for j in classes_i])
            N_remaining -= m_i
        return Episode(class_datasets, self.transforms)

    def get_episode_from_datasets(self, dataset_ids, classes_per_dataset):
        class_datasets=[]
        for dataset in dataset_ids:
            n_i = len(self.datasets_by_class[dataset])
            if classes_per_dataset >= n_i:
                class_datasets.append(self.datasets_by_class[dataset])
            else:
                classes_i = torch.multinomial(torch.ones(n_i), classes_per_dataset)
                class_datasets.append([self.datasets_by_class[dataset][j.item()] for j in classes_i])
        return Episode(class_datasets, self.transforms)
                

    def get_dataset(self, dataset_id):
        class_datasets = [self.datasets_by_class[dataset_id]]
        return Episode(class_datasets, self.transforms)



class Episode():
    def __init__(self, datasets, transforms):
        self.datasets = datasets
        self.sizes = [len(d) for d in datasets]
        self.N = sum(self.sizes)
        self.transforms = transforms

    def _class_to_dataset(self, class_id):
        assert class_id < self.N
        N=0
        for j, n_j in enumerate(self.sizes):
            if class_id >= N + n_j:
                N += n_j
            else:
                return j, class_id - N

    #@profile
    def _get_next(self, class_id, dataset_id=None):
        if dataset_id is None:
            dataset_id, class_id = self._class_to_dataset(class_id)
        try:
            sample_dic = next(self.datasets[dataset_id][class_id])
        except (StopIteration, TypeError) as e:
            self.datasets[dataset_id][class_id] = cycle_(self.datasets[dataset_id][class_id])
            sample_dic = next(self.datasets[dataset_id][class_id])
        return sample_dic

    #@profile
    def _generate_set_from_class(self, class_id, n_samples, dataset_id=None):
        set_data = []
        for i in range(n_samples):
            sample_dic = self._get_next(class_id, dataset_id=dataset_id)
            sample_dic = parse_record(sample_dic)
            transformed_image = self.transforms(sample_dic['image'])
            set_data.append(transformed_image)
        return set_data
    
    def _generate_set_from_dataset(self, dataset_id, n_samples):
        set_data = []
        for i in range(n_samples):
            class_id = torch.randint(len(self.datasets[dataset_id]), (1,)).item()
            sample_dic = self._get_next(class_id, dataset_id)
            sample_dic = parse_record(sample_dic)
            transformed_image = self.transforms(sample_dic['image'])
            set_data.append(transformed_image)
        return set_data

    def _generate(self, batch_size, set_size=(10,15), p_aligned=0.5, p_dataset=0.3, p_same=0.3, eval=False):
        dataset_level = (torch.rand(batch_size) < p_dataset)
        aligned = (torch.rand(batch_size) < p_aligned)
        same_dataset = (torch.rand(batch_size) < p_same)
        n_samples = torch.randint(*set_size, (1,)).item()
        X = []
        Y = []
        for j in range(batch_size):
            if dataset_level[j]:
                if aligned[j]:
                    dataset1 = torch.randint(len(self.datasets), (1,))
                    dataset2 = dataset1
                else:
                    dataset1, dataset2 = torch.multinomial(torch.ones(len(self.datasets)), 2)
                X_j = self._generate_set_from_dataset(dataset1.item(), n_samples)
                Y_j = self._generate_set_from_dataset(dataset2.item(), n_samples)
            else:
                if aligned[j]:
                    class1 = torch.randint(self.N, (1,))
                    class2 = class1
                    X_j = self._generate_set_from_class(class1.item(), n_samples)
                    Y_j = self._generate_set_from_class(class2.item(), n_samples)
                else:
                    if same_dataset[j]:
                        dataset1 = torch.randint(len(self.datasets), (1,)).item()
                        dataset2=dataset1
                        class1, class2 = torch.multinomial(torch.ones(self.sizes[dataset1]), 2)
                    else:
                        dataset1, dataset2 = torch.multinomial(torch.ones(len(self.datasets)), 2)
                        class1 = torch.randint(len(self.datasets[dataset1]), (1,))
                        class2 = torch.randint(len(self.datasets[dataset2]), (1,))
                    X_j = self._generate_set_from_class(class1.item(), n_samples, dataset_id=dataset1)
                    Y_j = self._generate_set_from_class(class2.item(), n_samples, dataset_id=dataset2)

            X.append(torch.stack(X_j, 0))
            Y.append(torch.stack(Y_j, 0))
        X = torch.stack(X, 0)
        Y = torch.stack(Y, 0)
        if eval:
            return (X,Y), aligned, (dataset_level, same_dataset)
        else:
            return (X,Y), aligned.float()

    def _generate_from_dataset(self, batch_size, dataset_id, set_size=(10,15), p_aligned=0.5):
        aligned = (torch.rand(batch_size) < p_aligned)
        n_samples = torch.randint(*set_size, (1,)).item()
        X = []
        Y = []
        for j in range(batch_size):
            if aligned[j]:
                class1 = torch.randint(self.sizes[dataset_id], (1,))
                class2 = class1
            else:
                class1, class2 = torch.multinomial(torch.ones(self.sizes[dataset_id]), 2)
            X_j = self._generate_set_from_class(class1.item(), n_samples, dataset_id=dataset_id)
            Y_j = self._generate_set_from_class(class2.item(), n_samples, dataset_id=dataset_id)
            X.append(torch.stack(X_j, 0))
            Y.append(torch.stack(Y_j, 0))
        X = torch.stack(X, 0)
        Y = torch.stack(Y, 0)
        return (X,Y), aligned.float()
        
    def compare_datasets(self, i, j, batch_size=1, set_size=(10,15)):
        n_samples = torch.randint(*set_size, (1,)).item()
        if batch_size == 1:
            X = torch.stack(self._generate_set_from_dataset(i, n_samples), dim=0)
            Y = torch.stack(self._generate_set_from_dataset(j, n_samples), dim=0)
        else:
            X = torch.stack([torch.stack(self._generate_set_from_dataset(i, n_samples), dim=0) for _ in range(batch_size)], dim=0)
            Y = torch.stack([torch.stack(self._generate_set_from_dataset(j, n_samples), dim=0) for _ in range(batch_size)], dim=0)
        return X, Y

    def compare_classes(self, d_i, d_j, c_i=None, c_j=None, batch_size=1, set_size=(10,15)):
        if d_i == d_j and c_i is None and c_j is None:
            c_i, c_j = torch.multinomial(torch.ones(self.sizes[d_i]), 2)
            c_i, c_j = c_i.item(), c_j.item()
        else:
            c_i = torch.randint(self.sizes(d_i), (1,)).item() if c_i is None else c_i
            c_j = torch.randint(self.sizes(d_j), (1,)).item() if c_j is None else c_j
        n_samples = torch.randint(*set_size, (1,)).item()
        if batch_size == 1:
            X = torch.stack(self._generate_set_from_class(c_i, n_samples, dataset_id=d_i), dim=0)
            Y = torch.stack(self._generate_set_from_class(c_j, n_samples, dataset_id=d_j), dim=0)
        else:
            X = torch.stack([torch.stack(self._generate_set_from_class(c_i, n_samples, dataset_id=d_i), dim=0) for _ in range(batch_size)], dim=0)
            Y = torch.stack([torch.stack(self._generate_set_from_class(c_j, n_samples, dataset_id=d_j), dim=0) for _ in range(batch_size)], dim=0)
        return X, Y

    def __call__(self, *args, **kwargs):
        if 'dataset_id' in kwargs:
            return self._generate_from_dataset(*args, **kwargs)
        else:
            return self._generate(*args, **kwargs)
            