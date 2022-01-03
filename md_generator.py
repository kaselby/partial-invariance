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
    def __init__(self, image_size=84, dataset_path=DATASET_ROOT, split=Split.TRAIN, device=torch.device('cpu')):
        self.split=split
        self.image_size = image_size
        self.device=device
        self.datasets_by_class = self._build_datasets()
        self.N = len(self.datasets_by_class)
        self.transforms = get_transforms(self.image_size, self.split)

    def _build_datasets(self, min_class_examples=20):
        datasets = []
        for dataset in ALL_DATASETS:
            dataset_spec = dataset_spec_lib.load_dataset_spec(os.path.join(DATASET_ROOT, dataset))
            reader = Reader(dataset_spec, self.split, False, 0) 
            class_datasets = reader.construct_class_datasets()
            if len(class_datasets) > 0:
                split_classes = dataset_spec.get_classes(self.split)
                filtered_class_datasets = [x for i, x in enumerate(class_datasets) if dataset_spec.get_total_images_per_class(split_classes[i]) >= min_class_examples]
                if len(class_datasets) > 0:
                    datasets.append(class_datasets)
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
        return Episode(class_datasets, self.transforms, device=self.device)

    '''
    def get_episode(self, n_classes, n_datasets):
        class_datasets=[]
        datasets = torch.multinomial(torch.ones(self.N), n_datasets) if n_datasets < self.N else torch.arange(self.N)
        n_datasets = len(datasets)
        classes_per_dataset = (torch.distributions.Dirichlet(torch.ones(n_datasets)/n_datasets).sample() * n_classes).round().long()
        for i in range(n_datasets):
            if classes_per_dataset[i] > 0:
                dataset_i = datasets[i].item()
                N_i = len(self.datasets_by_class[dataset_i])
                classes_i = torch.multinomial(torch.ones(N_i), classes_per_dataset[i].item())
                class_datasets += [self.datasets_by_class[dataset_i][j.item()] for j in classes_i]
        return Episode(class_datasets, self.transforms, p_aligned=self.p_aligned, device=self.device)
    '''

        

'''
    @profile
    def _get_next(self, dataset_id, class_id):
        try:
            sample_dic = next(self.datasets_by_class[dataset_id][class_id])
        except (StopIteration, TypeError) as e:
            self.datasets_by_class[dataset_id][class_id] = cycle_(self.datasets_by_class[dataset_id][class_id])
            sample_dic = next(self.datasets_by_class[dataset_id][class_id])

        return sample_dic

    @profile
    def _generate_set(self, dataset_id, class_id, n_samples):
        set_data = []
        for i in range(n_samples):
            sample_dic = self._get_next(dataset_id, class_id)
            sample_dic = parse_record(sample_dic)
            transformed_image = self.transforms(sample_dic['image'])
            set_data.append(transformed_image)
        return set_data

    @profile
    def _generate(self, batch_size, set_size=(10,15)):
        #def process_image(imgdict):
        #    return self.transforms(parse_record(imgdict)['image'])
        #def sample_dataset(dataset, data_class, n_samples):
        #    return [process_image()) for _ in range(n_samples)]

        aligned = (torch.rand(batch_size) < self.p_aligned)
        n_samples = torch.randint(*set_size, (1,)).item()
        X = []
        Y = []
        for j in range(batch_size):
            if aligned[j]:
                dataset1 = torch.randint(self.N, (1,)).item()
                dataset2 = dataset1
                class1 = torch.randint(len(self.datasets_by_class[dataset1]), (1,)).item()
                class2 = class1
            else:
                if torch.rand(1).item() < self.p_sameset:
                    dataset1 = torch.randint(self.N, (1,)).item()
                    dataset2 = dataset1
                    class1, class2 = torch.multinomial(torch.ones(len(self.datasets_by_class[dataset1])), 2)
                    class1, class2 = class1.item(), class2.item()
                else:
                    dataset1, dataset2 = torch.multinomial(torch.ones(self.N), 2)
                    dataset1, dataset2 = dataset1.item(), dataset2.item()
                    class1 = torch.randint(len(self.datasets_by_class[dataset1]), (1,)).item()
                    class2 = torch.randint(len(self.datasets_by_class[dataset2]), (1,)).item()
            X_j = self._generate_set(dataset1, class1, n_samples)
            Y_j = self._generate_set(dataset2, class2, n_samples)
            X.append(torch.stack(X_j, 0))
            Y.append(torch.stack(Y_j, 0))
        X = torch.stack(X, 0)
        Y = torch.stack(Y, 0)
        return (X.to(self.device),Y.to(self.device)), aligned.to(self.device).float()

    def __call__(self, *args, **kwargs):
        return self._generate(*args, **kwargs)
'''


class Episode():
    def __init__(self, datasets, transforms, device=torch.device('cpu')):
        self.datasets = datasets
        self.sizes = [len(d) for d in datasets]
        self.N = sum(self.sizes)
        self.transforms = transforms
        self.device = device

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

    def _generate(self, batch_size, set_size=(10,15), p_aligned=0.5, p_dataset=0.3, p_same=0.5, eval=False):
        dataset_level = (torch.rand(batch_size) < p_aligned)
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
                        class1, class2 = torch.multinomial(torch.ones(self.sizes[dataset]), 2)
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
            return (X.to(self.device),Y.to(self.device)), aligned.to(self.device), (dataset_level, same_dataset)
        else:
            return (X.to(self.device),Y.to(self.device)), aligned.to(self.device).float()

    def __call__(self, *args, **kwargs):
        return self._generate(*args, **kwargs)
            