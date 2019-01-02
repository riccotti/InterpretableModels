import sys
import subprocess


dataset_size = {
    'adult': 48842,
    'anneal': 898,
    'census': 299285,
    'clean1': 476,
    'clean2': 6598,
    'coil2000': 9822,
    'cover': 581012,
    'credit': 1000,
    'sonar': 208,
    'soybean': 683
}


def main():

    dataset = sys.argv[1]

    args = ['python', 'evaluate_stability_tree2.py',
            '-d', dataset + '_yadt',
            '-m', "('DT', 'yadt')",
            '-dp', '../datasets/',
            '-mp', '../models_tree/',
            '-rp', '../results_tree/'
            ]

    print('yadt depth None')
    subprocess.run(args)

    args = ['python', 'evaluate_stability_tree2.py',
            '-d', dataset+'_yadt',
            '-m', "('DT', 'yadt')",
            '-dp', '../datasets/',
            '-mp', '../models_tree/',
            '-rp', '../results_tree/',
            '-depth', '5'
            ]

    print('yadt depth 5')
    subprocess.run(args)

    
if __name__ == "__main__":
    main()
