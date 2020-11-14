import numpy as np
import os
import subprocess
from multiprocessing import Pool

def read_accuracy():
    with open('accuracy.txt') as f:
        accuracy = f.readlines()

    accuracy = [float(x.split()[1]) for x in accuracy[1:]]
    accuracy = np.array(accuracy)
    return np.mean(accuracy)


def run_test(name, test, runs=1, num_charts=10, use_text=True, train=None):
    if not os.path.exists(name):
        os.makedirs(name)

    os.chdir(name)
    args = ['../env/bin/python', '../classifier.py', test, runs, num_charts, use_text]

    if train:
        args.append('--train')
        args.append(train)

    # make sure strings
    args = list(map(str, args))

    # call process and write output to a file
    with open('out', 'w') as f:
        subprocess.call(args, stdout=f)

    a = read_accuracy()
    os.chdir('..')
    print("finished " + name)
    return a


def run_test_from_list(l):
    name = l[0]
    test = l[1]
    if len(l) > 2 and l[2] is not None:
        runs = l[2]
    else:
        runs = 1
    if len(l) > 3 and l[3] is not None:
        num_charts = l[3]
    else:
        num_charts = 10
    if len(l) > 4 and l[4] is not None:
        use_text = l[4]
    else:
        use_text = False
    if len(l) > 5 and l[5] is not None:
        train = l[5]
    else:
        train = None
    return name + ": " + str(run_test(name, test, runs=runs, num_charts=num_charts, use_text=use_text, train=train))


if __name__ == '__main__':
    tests = [
                # Running on full sets (10 runs)
                ['d3 full', 'd3', 10],
                ['plotly full', 'plotly', 10],
                ['chartblocks full', 'chartblocks', 10],
                ['graphiq full', 'graphiq', 10],
                ['fusion full', 'fusion', 10],

                # 5 Fold Cross-validation with subsets 10 max number charts (40 runs)
                ['d3 10 max charts', 'd3', 40, 10, True],
                ['plotly 10 max charts', 'plotly', 40, 10, True],
                ['chartblocks 10 max charts', 'chartblocks', 40, 10, True],
                ['graphiq 10 max charts', 'graphiq', 40, 10, True],
                ['fusion 10 max charts', 'fusion', 40, 10, True],

                # 5 Fold Cross-validation with subsets 20 max number charts (40 runs):
                ['d3 20 max charts', 'd3', 40, 20, True],
                ['plotly 20 max charts', 'plotly', 40, 20, True],
                ['chartblocks 20 max charts', 'chartblocks', 40, 20, True],
                ['graphiq 20 max charts', 'graphiq', 40, 20, True],
                ['fusion 20 max charts', 'fusion', 40, 20, True],

                # Train on one, test on others (10 runs)
                ['train d3, test plotly', 'plotly', 10, 20, True, 'd3'],
                ['train d3, test chartblocks', 'chartblocks', 10, 20, True, 'd3'],
                ['train d3, test graphiq', 'graphiq', 10, 20, True, 'd3'],
                ['train d3, test fusion', 'fusion', 10, 20, True, 'd3'],

                ['train plotly, test d3', 'd3', 10, 20, True, 'plotly'],
                ['train plotly, test chartblocks', 'chartblocks', 10, 20, True, 'plotly'],
                ['train plotly, test graphiq', 'graphiq', 10, 20, True, 'plotly'],
                ['train plotly, test fusion', 'fusion', 10, 20, True, 'plotly'],

                ['train chartblocks, test d3', 'd3', 10, 20, True, 'chartblocks'],
                ['train chartblocks, test plotly', 'plotly', 10, 20, True, 'chartblocks'],
                ['train chartblocks, test graphiq', 'graphiq', 10, 20, True, 'chartblocks'],
                ['train chartblocks, test fusion', 'fusion', 10, 20, True, 'chartblocks'],

                ['train graphiq, test d3', 'd3', 10, 20, True, 'graphiq'],
                ['train graphiq, test plotly', 'plotly', 10, 20, True, 'graphiq'],
                ['train graphiq, test chartblocks', 'chartblocks', 10, 20, True, 'graphiq'],
                ['train graphiq, test fusion', 'fusion', 10, 20, True, 'graphiq'],

                ['train fusion, test d3', 'd3', 10, 20, True, 'fusion'],
                ['train fusion, test plotly', 'plotly', 10, 20, True, 'fusion'],
                ['train fusion, test chartblocks', 'chartblocks', 10, 20, True, 'fusion'],
                ['train fusion, test graphiq', 'graphiq', 10, 20, True, 'fusion'],

                # train sets of 2
                ['test d3,plotly', 'd3,plotly', 10],
                ['test d3,chartblocks', 'd3,chartblocks', 10],
                ['test d3,graphiq', 'd3,graphiq', 10],
                ['test d3,fusion', 'd3,fusion', 10],
                ['test plotly,chartblocks', 'plotly,chartblocks', 10],
                ['test plotly,graphiq', 'plotly,graphiq', 10],
                ['test plotly,fusion', 'plotly,fusion', 10],
                ['test chartblocks,graphiq', 'chartblocks,graphiq', 10],
                ['test chartblocks,fusion', 'chartblocks,fusion', 10],
                ['test graphiq,fusion', 'graphiq,fusion', 10],

                # train sets of 3
                ['test d3,plotly,chartblocks', 'd3,plotly,chartblocks', 10],
                ['test d3,plotly,graphiq', 'd3,plotly,graphiq', 10],
                ['test d3,plotly,fusion', 'd3,plotly,fusion', 10],
                ['test d3,chartblocks,graphiq', 'd3,chartblocks,graphiq', 10],
                ['test d3,chartblocks,fusion', 'd3,chartblocks,fusion', 10],
                ['test d3,graphiq,fusion', 'd3,graphiq,fusion', 10],
                ['test plotly,chartblocks,graphiq', 'plotly,chartblocks,graphiq', 10],
                ['test plotly,chartblocks,fusion', 'plotly,chartblocks,fusion', 10],
                ['test plotly,graphiq,fusion', 'plotly,graphiq,fusion', 10],
                ['test chartblocks,graphiq,fusion', 'chartblocks,graphiq,fusion', 10],

                # train sets of 4
                ['test d3,plotly,chartblocks,graphiq', 'd3,plotly,chartblocks,graphiq', 10],
                ['test d3,plotly,chartblocks,fusion', 'd3,plotly,chartblocks,fusion', 10],
                ['test d3,plotly,graphiq,fusion', 'd3,plotly,graphiq,fusion', 10],
                ['test d3,chartblocks,graphiq,fusion', 'd3,chartblocks,graphiq,fusion', 10],
                ['test plotly,chartblocks,graphiq,fusion', 'plotly,chartblocks,graphiq,fusion', 10],

                # train all
                ['test all 5', 'd3,plotly,chartblocks,graphiq,fusion', 10]
            ]

    p = Pool(3, maxtasksperchild=1)

    results = p.map(run_test_from_list, tests, chunksize=1)

    with open('test_results.txt', 'w') as f:
        for result in results:
            print(result)
            f.write(result + '\n')
