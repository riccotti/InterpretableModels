

def prepare_iris_dataset(dataset_name, dataset_path):
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


datasets = {
    'iris': prepare_iris_dataset
}


def get_dataset(dataset_name, dataset_path):

    return datasets[dataset_name](dataset_name, dataset_path)
