import argparse
import json
import os
import random
import time
import traceback

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from d3_feature_extractor import extract

'''
creates and RUNS the decision tree.  Takes n random samples of each chart type for a stratified k-fold evaluation, where n is specified by the NUM_CHARTS global variable.
It prints the average accuracy for all lables and RUNS and writes them to 'accuracy.txt'.
For incorrectly classified charts, it writes the features dictionary, label returned by the classifier, and its actual label to 'wrong.txt'.
'''

features = None
badCharts = None

# mapping of integers to the chart type they represent
symbol_dict = {
    1: "line",
    2: "scatter",
    4: "bar",
    19: "geographic_map",
    35: "graph",
    14: "chord",
    10: "bubble",
    37: "parallel_coordinates",
    13: "sankey",
    9: "box",
    16: "area",
    31: "stream_graph",
    7: "heat_map",
    15: "radial",
    33: "hexabin",
    38: "sunburst",
    22: "treemap",
    40: "voronoi",
    18: "donut",
    39: "waffle",
    41: "word_cloud",
    29: "pie"
}

# Some labels in the dataset use the name instead of the integer value (e.g. "scatter" instead of 2)
inv_symbol_dict = {v: k for k, v in symbol_dict.items()}

text_features = [
    "text_word_count",
    "text_max_font_size",
    "text_min_font_size",
    "text_var_font_size",
    "text_unique_font_size_count",
    "text_unique_x_count",
    "text_unique_y_count"
]

NUM_CHARTS = 10  # number of charts to sample per vis type
MAX_CHARTS = 400  # maximum number of charts to include per vis type
RUNS = 1  # number of times to repeat the experiment
USE_TEXT = True

# svg_path = os.path.join(data_path,"charts")
parser = argparse.ArgumentParser(prog='Beagle Classifier')
parser.add_argument('--t', help="Tells you the set that it is trained on")
parser.add_argument('args', nargs="+", help="Normal args that would have just been collected before")
sysargs = parser.parse_args()

args = sysargs.args
trained = sysargs.t

'''
Paths that are specific to data that exists in modis already
'''
collection = args[0]
data_path = "%s/charts/" % collection
urls_file_path = "%s/urls.txt" % collection
output_file = os.path.join(data_path, "features.txt")
# for automatically ignoring bad charts
badfile = os.path.join(data_path, "bad.txt")
images_path = "%s/images/" % collection
svg_path = "%s/charts/" % collection

if len(args) > 1:
    RUNS = int(args[1])  # number of times to repeat the experiment

if len(args) > 2:
    NUM_CHARTS = int(args[2])  # number of charts to sample per vis type

if len(args) > 3:
    USE_TEXT = args[3].lower() == "true"
    print("using text? ", USE_TEXT)


# Convert numpy data types to primitive Python types, used in json.dumps()
# See: https://stackoverflow.com/a/57915246/15507541
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_features():
    global features
    if not os.path.isfile(output_file):
        print("creating features from scratch")
        create_and_save_features()
    else:
        print("loading features file")
        print(output_file)
        # file = open(output_file, 'rb')
        # features = pickle.load(file)
        with open(output_file, 'r') as f:
            # features = pickle.load(output_file)
            # data = f.read()
            # f = f.decode('utf-8').replace('\0', '')
            features = json.load(f)


def create_and_save_features():
    global features
    global badCharts
    features = {}
    badCharts = {}
    if os.path.isfile(badfile):
        with open(badfile) as f:
            badCharts = json.load(f)
    urls_file = open(urls_file_path, 'r')
    counter = 0
    for line in urls_file:
        flags = line.strip().rstrip('\n').rstrip('\r').split(" ")
        chart = flags[0]
        if chart in badCharts:
            continue
        if not os.path.exists(os.path.join(svg_path, chart, "svg.txt")):
            badCharts[chart] = True
            continue
        multiple_labels = flags[2].split(",")
        # Some dataset labels use the string value instead of integer, handle those cases by converting back
        try:
            if not multiple_labels[0].lstrip('-').isdigit():
                label = -1 if multiple_labels[0] == 'other' else int(inv_symbol_dict[multiple_labels[0]])
            else:
                label = int(multiple_labels[0])
        except KeyError:
            badCharts[chart] = True
            continue
        # ignore charts with bad images
        if label not in symbol_dict:  # ignore unsupported chart types
            badCharts[chart] = True
            continue
        if "i" in flags:
            badCharts[chart] = True
            continue
        if not testImage(os.path.join(images_path, chart + ".png")):
            badCharts[chart] = True
            continue
        feature_dict = extract(os.path.join(svg_path, chart, "svg.txt"))
        print("Features created for chart ID-%s, n features = %d" % (chart, len(feature_dict)))
        if isinstance(feature_dict, str):
            print("wrong output")
            badCharts[chart] = True
            continue
        features[chart] = feature_dict
        print("Line num: %d" % counter)
        counter += 1
    print(features)
    output_json = open(output_file, 'w')
    json.dump(features, output_json, cls=NpEncoder)
    with open(badfile, 'w') as f:
        json.dump(badCharts, f)
    output_json.close()


# used to see if we have a valid image
def testImage(imagepath):
    try:
        Image.open(imagepath)
        return True
    except:
        pass
        return False  # didn't work


# goes over all samples and filters for only valid vis types
# each vis type has a list of (chart,url) pairs
# only call once
def organize_samples(types_lists, secondary_labels):
    print("organizing samples...")
    urls_file = open(urls_file_path, 'r')
    start = time.time()
    for line in urls_file:
        line_list = line.rstrip("\n").rstrip("\r").split(" ")
        chart = line_list[0]
        if chart not in features:
            continue
        secondary_label = None
        url = line_list[1]
        multiple_labels = line_list[2].split(",")
        label = int(multiple_labels[0])
        if label not in symbol_dict:
            continue
        if len(multiple_labels) > 1:
            secondary_label = multiple_labels[1:]
        types_lists[label].append((chart, url))
        if secondary_label is not None:
            secondary_labels[chart] = secondary_label


# select subsets from each vis type, and extract feature for these types
# call this function for each new run
def select_subsets(feature_dicts, labels, urls, types_lists):
    print("\tselecting sample subsets (" + str(NUM_CHARTS) + " samples per chart type) and extracting features...")
    for label in types_lists:  # for each valid vis type
        if len(types_lists[label]) == 0:
            continue
        if len(types_lists[label]) < NUM_CHARTS:  # not enough samples, repeat samples evenly
            diff = int(NUM_CHARTS / len(types_lists[label]))
            remainder = NUM_CHARTS - diff * len(types_lists[label])
            chosenSamples = list(types_lists[label]) * diff
            for i in range(remainder):
                chosenSamples.append(types_lists[label][i])
        else:  # enough samples
            chosenSamples = random.sample(types_lists[label], NUM_CHARTS)
        print("\t\textracting features for label " + str(label) + " (" + symbol_dict[label] + ")...")
        for chart, url in chosenSamples:
            try:
                if chart in features:
                    print("CHART ID: %s" % chart)
                    feature_dict = features[chart]
                else:
                    continue
                if isinstance(feature_dict, str):
                    continue
                if not USE_TEXT:
                    for k in text_features:
                        if k in feature_dict:
                            feature_dict.pop(k)
                feature_dicts.append(feature_dict)
                labels.append(label)
                urls.append((chart, url))
            except Exception as e:
                print(e)
                traceback.print_exc()
                pass


# limits number of charts included to at most 400 per label
def select_subsets_no_repeats(feature_dicts, labels, urls, types_lists):
    print("\tselecting all samples and extracting features...")
    for label in list(types_lists.keys()):  # for each valid vis type
        print("\t\textracting features for label " + str(label) + " (" + symbol_dict[label] + ")...")
        chosenSamples = types_lists[label]
        if len(chosenSamples) > MAX_CHARTS:
            chosenSamples = random.sample(types_lists[label], MAX_CHARTS)
        for chart, url in chosenSamples:
            try:
                if chart in features:
                    feature_dict = features[chart]
                if isinstance(feature_dict, str):
                    continue
                if not USE_TEXT:
                    for k in text_features:
                        if k in feature_dict:
                            feature_dict.pop(k)
                feature_dicts.append(feature_dict)
                labels.append(label)
                urls.append((chart, url))
            except Exception as e:
                print(e)
                traceback.print_exc()
                pass


# performs the cross validation for the chosen samples
def cross_validation(feature_dicts, labels, secondary_labels, urls):
    vec = DictVectorizer(sparse=False)
    scaler = StandardScaler()
    print("\tcreating features array...")

    print("\tperforming stratified k-fold...")

    skf = StratifiedKFold(n_splits=5)
    features_array = scaler.fit_transform(vec.fit_transform(feature_dicts))
    clf = RandomForestClassifier(n_estimators=14)

    # Save trained Beagle model on all available data, prior to cross validation
    train_save_beagle_model(features_array, labels, clf, vec, scaler)

    correct_count = {}
    total_count = {}
    wrong = {}
    result = {}
    for train_index, test_index in skf.split(features_array, labels):
        training_dicts = [features_array[t] for t in train_index]
        training_labels = [labels[t] for t in train_index]
        testing_points = [features_array[t] for t in test_index]
        testing_labels = [labels[t] for t in test_index]
        clf.fit(training_dicts, training_labels)

        for i in range(len(testing_points)):
            label = testing_labels[i]
            for d in (correct_count, total_count):
                if label not in list(d.keys()):
                    d[label] = 0
            total_count[label] += 1
            prediction = clf.predict(np.array(testing_points[i]).reshape(1, -1))[0]
            if prediction == label:
                correct_count[label] += 1
            else:
                failed = False
                folder = urls[test_index[i]][0]
                if folder in list(secondary_labels.keys()):
                    if str(prediction) in secondary_labels[folder]:
                        correct_count[label] += 1
                    else:
                        failed = True
                else:
                    failed = True
                if failed:
                    real_label = symbol_dict[label]
                    if real_label not in wrong:
                        wrong[real_label] = []
                    wrong[real_label].append({
                        'feature_dict': feature_dicts[test_index[i]],
                        'url': urls[test_index[i]],
                        "predicted": symbol_dict[np.asscalar(prediction)],
                        "real": symbol_dict[label]
                    })

    save_model(clf, "model_cv_d3.pkl")

    for k in total_count:  # for each label
        result[symbol_dict[k]] = 1.0 * correct_count[k] / total_count[k]
    return {"accuracy": result, "wrong": wrong, "correct": correct_count, "total": total_count}


def accuracy_across_runs(runResults):
    final_correct = 0
    final_total = 0
    final_accuracy_per_label = {}
    final_wrong = {}
    for runResult in runResults:
        accuracy = runResult['accuracy']
        wrong = runResult['wrong']
        correct_count = runResult['correct']
        total_count = runResult['total']
        for label in accuracy:
            if label not in final_accuracy_per_label:
                final_accuracy_per_label[label] = []
            final_accuracy_per_label[label].append(accuracy[label])
        print(correct_count, total_count)
        for label in correct_count:
            final_correct += correct_count[label]
        for label in total_count:
            final_total += total_count[label]
        for label in wrong:
            if label not in final_wrong:
                final_wrong[label] = []
            final_wrong[label].extend(wrong[label])
    print("average accuracy across all RUNS:")
    avg_accuracy = []
    with open("accuracy.txt", "w") as myfile:
        myfile.write("\t".join(['label', 'accuracy']) + "\n")
        for label in sorted(final_accuracy_per_label):
            final_average = np.mean(final_accuracy_per_label[label])
            print(label + "\t" + str(final_average) + "\t", final_accuracy_per_label[label])
            avg_accuracy.append(final_average)
            myfile.write("\t".join([label, str(final_average)]) + "\n")
    print("average:")
    print(np.mean(np.array(avg_accuracy)))

    print("overall:")
    print(1.0 * final_correct / final_total)

    with open("wrong.txt", "w") as myfile:
        myfile.write("\t".join(['visid', 'real', 'predicted', 'url', 'feature_dict']) + "\n")
        for label in sorted(final_wrong):
            for w in final_wrong[label]:
                myfile.write("\t".join(
                    [w['url'][0], w['real'], w['predicted'], w['url'][1],
                     json.dumps(w['feature_dict'], cls=NpEncoder)]) + "\n")
    wrong_counts = {}
    for label in sorted(final_wrong):
        for w in final_wrong[label]:
            real = w['real']
            predicted = w['predicted']
            if wrong_counts.get(real):
                if wrong_counts[real].get(predicted):
                    wrong_counts[real][predicted] += 1
                else:
                    wrong_counts[real][predicted] = 1
            else:
                wrong_counts[real] = {predicted: 1}
    wrong_counts_file = open("wrong_counts.txt", 'w')
    for chart in list(wrong_counts.keys()):
        wrong_counts_file.write(chart + ":\n")
        sorted_items = sorted(list(wrong_counts[chart].items()), key=lambda x: x[1])
        for chart_type, count in sorted_items:
            wrong_counts_file.write(chart_type + ": " + str(count) + '\n')
        wrong_counts_file.write("\n")
    wrong_counts_file.close()


# used to tell us how many valid charts there are for each chart type
def print_chart_stats(types_lists):
    print("chart stats:")
    outp = {}
    for l in types_lists:
        outp[symbol_dict[l]] = len(types_lists[l])
    print(outp)


def save_model(clf, filename):
    """
    Saves the trained model for future use in a pickle.
    clf: the model
    filename: name of the file to save
    """
    joblib.dump(clf, filename)


def train_save_beagle_model(features_array, labels, clf, vec, scaler):
    clf.fit(features_array, labels)
    save_model(clf, "model_d3.pkl")
    save_object(vec, "vectorizer_d3.pkl")
    save_object(scaler, "scaler_d3.pkl")


def save_object(obj, filename):
    joblib.dump(obj, filename)


def reload_model(filename):
    """
    @return the model that was saved to the pickle at filename
    """
    return joblib.load(filename)


def main():
    # holds all available samples collected from urls.txt, separated by vis type
    # valid vis types only
    types_lists = {}
    for k in symbol_dict:
        types_lists[k] = []

    # if vis sample has more than one label, put the second label in here
    secondary_labels = {}

    total_time = []
    accuracy = []

    s = time.time()
    load_features()  # build the feature dicts, if they don't exist yet
    organize_samples(types_lists, secondary_labels)  # only call once
    print(secondary_labels)
    total_time.append(time.time() - s)
    print("running experiment " + str(RUNS) + " time(s)...")
    zero_features = []
    for run_index in range(RUNS):
        print("executing run " + str(run_index) + "...")
        feature_dicts = []
        labels = []
        urls = []
        s = time.time()
        if NUM_CHARTS == -1:
            select_subsets_no_repeats(feature_dicts, labels, urls, types_lists)
        else:
            select_subsets(feature_dicts, labels, urls, types_lists)
        total_time.append(time.time() - s)
        result_dict = cross_validation(feature_dicts, labels, secondary_labels, urls)
        accuracy.append(result_dict)
    print("average extraction time:", np.sum(total_time) / RUNS)
    accuracy_across_runs(accuracy)
    print_chart_stats(types_lists)


if __name__ == "__main__":
    main()
