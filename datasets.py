



def load_coco_data(imgdir, anndir):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    train_dataset = CocoCaptions(root=os.path.join(imgdir, "train2014"), annFile=os.path.join(anndir, "captions_train2014.json"), transform=transforms)
    val_dataset = CocoCaptions(root=os.path.join(imgdir, "val2014"), annFile=os.path.join(anndir, "captions_val2014.json"), transform=transforms)

    return train_dataset, train_dataset, val_dataset

def load_flickr_data(imgdir, annfile, split_file):
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    dataset = Flickr30k(root=os.path.join(imgdir), ann_file=annfile, transform=transforms)

    with open(split_file, 'r') as f:
        splits_dict=json.load(f)
    splits = {'train':[], 'val':[], 'test':[]}
    for i in range(len(splits_dict['images'])):
        img_dict = splits_dict['images'][i]
        splits[img_dict["split"]].append(img_dict["imgid"])

    train_dataset = Subset(dataset, splits["train"])
    val_dataset = Subset(dataset, splits["val"])
    test_dataset = Subset(dataset, splits["test"])

    return train_dataset, val_dataset, test_dataset


def load_omniglot(root_folder="./data"):
    '''
    train_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=True
    )

    test_dataset = torchvision.datasets.Omniglot(
        root=root_folder, download=True, transform=torchvision.transforms.ToTensor(), background=False
    )
    '''
    transforms = torchvision.transforms.ToTensor()
    train_dataset, = ModifiedOmniglotDataset.splits(root_folder, -1, transform=transforms, img_dir="images_background")
    val_dataset, test_dataset = ModifiedOmniglotDataset.splits(root_folder, 5, -1, transform=transforms, img_dir="images_evaluation")
    

    return train_dataset, val_dataset, test_dataset


def load_mnist(root_folder="./data"):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=root_folder, download=True, transform=transform, train=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root_folder, download=True, transform=transform, train=False
    )

    return train_dataset, test_dataset

def load_cifar(root_folder="./data"):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.1642, 0.1496, 0.1728))
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=root_folder, download=True, transform=transform, train=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_folder, download=True, transform=transform, train=False
    )

    return train_dataset, test_dataset