from dataset import get_dataset


def process_dataset(name):
    dataset_config = {'name': name + 'Dataset', 'path': 'data/' + name,
                      'device': 'cpu', 'train_ratio': 0.8, 'min_inter': 10}
    dataset = get_dataset(dataset_config)
    dataset.output_dataset('data/' + name + '/time')


def main():
    process_dataset('MIND')
    # Amazon Users 21781, Items 10650, Average number of interactions 17.989, Total interactions 391828.0
    # Mind Users 22172, Items 9660, Average number of interactions 30.302, Total interactions 671848.0

if __name__ == '__main__':
    main()