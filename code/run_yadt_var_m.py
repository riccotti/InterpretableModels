import sys
import subprocess


dataset_size = {
    'adult': 48842,
    'anneal':898,
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

    m_limit = dataset_size[dataset]//2
    m_list = list()
    index = 1
    while True:
        m = 2**index
        if m > m_limit:
            break
        m_list.append(m)
        index += 1

    for m in m_list:
        args = ['python', 'evaluate_stability_tree.py',
                '-d', dataset+'_yadt',
                '-m', "('DT', 'yadt')",
                '-dp', '../datasets/',
                '-mp', '../models_tree/',
                '-rp', '../results_tree/',
                #'-v',
                '-y', str(m)
                ]

        print('yadt m %s' % m)
        subprocess.run(args)


if __name__ == "__main__":
    main()
