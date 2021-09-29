import argparse
import json
import os
import random
import time
import traceback

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from d3_feature_extractor import extract
from utility import reverse_dict

# mapping of integers to the chart type they represent
symbol_dict = {
    1: "line", 2: "scatter", 4: "bar", 19: "geographic_map", 35: "graph",
    14: "chord", 10: "bubble", 37: "parallel_coordinates", 13: "sankey",
    9: "box", 16: "area", 31: "stream_graph", 7: "heat_map",
    15: "radial", 33: "hexabin", 38: "sunburst", 22: "treemap",
    40: "voronoi", 18: "donut", 39: "waffle", 41: "word_cloud", 29: "pie"
}
num_dict = reverse_dict(symbol_dict)

text_features = [
    "text_word_count", "text_max_font_size", "text_min_font_size", "text_var_font_size",
    "text_unique_font_size_count", "text_unique_x_count", "text_unique_y_count"
]


class BeagleClassifier(object):
    """
    Object that contains the classifier for Beagle.

    NOTE: When initializing args requires at least the collections name
    """

    def __init__(self, args, training=None):
        # set presets
        self.num_charts = 10  # number of charts to sample per vis type
        self.max_charts = 400  # maximum number of charts to include per vis type
        self.runs = 1  # number of times to repeat the experiment
        self.use_text = True

        # set paths for files
        self.collection = args[0].split(',')
        self.training = training.split(',') if training else None
        self.features = {}
        self.train_features = {}

        # assumes all data in this directory, TODO make more general
        self.base_path = "/data/scidb/000/2/learnvis_data/"

        # override defaults based on args
        if len(args) > 1:
            self.runs = int(args[1])  # number of times to repeat the experiment

        if len(args) > 2:
            self.num_charts = int(args[2])  # number of charts to sample per vis type

        if len(args) > 3:
            self.use_text = args[3].lower() == "true"
            print("using text? ", self.use_text)

    def load_features(self):
        """
        Loads the features from the features.txt file if it exists. Otherwise
        it extracts the features from the SVGs and creates a features.txt file
        that holds all of the features.
        """
        # handle base set (test set if training exists)
        for collection in self.collection:
            output_file = os.path.join(self.base_path, collection, "charts", "features.txt")

            if not os.path.isfile(output_file):
                print("creating features from scratch")
                self.features.update(self.create_features(collection))
            else:
                print("loading features file")
                with open(output_file) as f:
                    self.features.update(json.load(f))

        # handle training set if specified
        if self.training is not None:
            for collection in self.training:
                output_file = os.path.join(self.base_path, collection, "charts", "features.txt")

                if not os.path.isfile(output_file):
                    print("creating features from scratch for training")
                    self.train_features.update(self.create_features(collection))
                else:
                    print("loading features file for training set")
                    with open(output_file) as f:
                        self.train_features.update(json.load(f))

    def create_features(self, collection):
        """
        Creates the features for all of the SVGs in the collection and
        stores the file in self.output_file which is a json of features
        """
        features = {}
        badCharts = {}

        # files for reading/storing info
        data_path = os.path.join(self.base_path, collection, "charts")
        urls_file_path = os.path.join(self.base_path, collection, "urls.txt")
        images_path = os.path.join(self.base_path, collection, "images")
        output_file = os.path.join(data_path, "features.txt")
        badfile = os.path.join(data_path, "bad.txt")

        if os.path.isfile(badfile):
            with open(badfile) as f:
                badCharts = json.load(f)
        urls_file = open(urls_file_path, 'r')
        for line in urls_file:
            flags = line.strip().rstrip('\n').rstrip('\r').split(" ")
            chart = collection + flags[0]
            chartsubname = flags[0]
            if chart in badCharts:
                continue
            if not os.path.exists(os.path.join(data_path, chartsubname, "svg.txt")):
                badCharts[chart] = True
                continue
            multiple_labels = flags[2].split(",")

            for i, label in enumerate(multiple_labels):
                if label in num_dict:
                    multiple_labels[i] = num_dict[label]
            try:
                label = int(multiple_labels[0])
            except:
                print("bad label", multiple_labels[0])
                continue
            # ignore charts with bad images
            if label not in symbol_dict:  # ignore unsupported chart types
                badCharts[chart] = True
                continue
            if "i" in flags:
                badCharts[chart] = True
                continue
            if not self.test_image(os.path.join(images_path, chartsubname + ".png")):
                badCharts[chart] = True
                continue
            feature_dict = extract(os.path.join(data_path, chartsubname, "svg.txt"))
            if isinstance(feature_dict, str):
                print("wrong output")
                badCharts[chart] = True
                continue
            features[chart] = feature_dict
        output_json = open(output_file, 'w')
        json.dump(features, output_json)
        with open(badfile, 'w') as f:
            json.dump(badCharts, f)
        output_json.close()
        return features

    def test_image(self, image_path):
        """
        Tests if the file at image_path is an image file
        """
        try:
            Image.open(image_path)
            return True
        except:
            pass
        return False  # didn't work

    def organize_samples(self, collection, types_lists, secondary_labels):
        """
        Goes over all samples and filters for only valid vis types
        each vis type has a list of (chart,url) pairs.
        Modifies types_lists with additions of charts of that type.
        Only gets called once!

        collection has to either be self.collection or self.training
        """
        print("organizing samples...")
        urls_file_path = os.path.join(self.base_path, collection, "urls.txt")
        urls_file = open(urls_file_path, 'r')
        try:
            features = self.features
        except:
            # is probably from training, so just pass for now
            pass
        if self.training and collection in self.training:
            features = self.train_features

        for line in urls_file:
            line_list = line.rstrip("\n").rstrip("\r").split(" ")
            chart = collection + line_list[0]
            if chart not in features:
                continue
            secondary_label = None
            url = line_list[1]
            multiple_labels = line_list[2].split(",")
            try:
                label = int(multiple_labels[0])
            except:
                try:
                    multiple_labels = [num_dict[x] for x in multiple_labels]
                    label = int(multiple_labels[0])
                except:
                    # label not a label we do
                    continue
            if label not in symbol_dict:
                continue
            if len(multiple_labels) > 1:
                secondary_label = multiple_labels[1:]
            types_lists[label].append((chart, url))
            if secondary_label is not None:
                secondary_labels[chart] = secondary_label

    def select_subsets(self, feature_dicts, labels, urls, types_lists, repeat=True, training=False):
        """
        Selects subsets from each vis type and extracts the features for this type.
        Samples a vis type if has >num_charts amount
        Gets called on each new run.

        If repeat is false, then it will not repeat when there are <num_charts
        """
        features = self.features if not training else self.train_features

        print("\tselecting sample subsets (" + str(
            self.num_charts) + " samples per chart type) and extracting features...")
        for label in types_lists:  # for each valid vis type
            chosenSamples = types_lists[label]

            if repeat:
                if len(types_lists[label]) == 0:
                    continue
                # not enough samples, repeat samples evenly
                if len(types_lists[label]) < self.num_charts:
                    diff = self.num_charts / len(types_lists[label])
                    remainder = self.num_charts - diff * len(types_lists[label])
                    chosenSamples = list(types_lists[label]) * diff
                    for i in range(remainder):
                        chosenSamples.append(types_lists[label][i])
                # enough samples
                else:
                    chosenSamples = random.sample(types_lists[label], self.num_charts)
            else:
                if len(chosenSamples) > self.max_charts:
                    chosenSamples = random.sample(types_lists[label], self.max_charts)

            print("\t\textracting features for label " + str(label) + " (" + symbol_dict[label] + ")...")
            for chart, url in chosenSamples:
                try:
                    if chart in features:
                        feature_dict = features[chart]
                    else:
                        continue

                    if isinstance(feature_dict, str):
                        continue
                    if not self.use_text:
                        for k in text_features:
                            if k in feature_dict:
                                feature_dict.pop(k)
                    # for k in features_to_ignore:
                    #   if k in feature_dict:
                    #     print k
                    #     feature_dict.pop(k)
                    feature_dicts.append(feature_dict)
                    labels.append(label)
                    urls.append((chart, url))
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    pass

    def do_cross_validation(self, feature_dicts, labels, secondary_labels, urls):
        """
        Performs the cross validation for the chosen samples
        """
        vec = DictVectorizer(sparse=False)
        scaler = StandardScaler()
        print("\tcreating features array...")

        print("\tperforming stratified k-fold...")
        skf = StratifiedKFold(labels, n_folds=5)
        features_array = scaler.fit_transform(vec.fit_transform(feature_dicts))
        # clf = RandomForestClassifier(n_estimators=14)
        correct_count = {}
        total_count = {}
        wrong = {}
        result = {}
        for train_index, test_index in skf:
            training_dicts = [features_array[t] for t in train_index]
            training_labels = [labels[t] for t in train_index]
            testing_points = [features_array[t] for t in test_index]
            testing_labels = [labels[t] for t in test_index]

            clf = RandomForestClassifier(n_estimators=14)
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

        for k in total_count:  # for each label
            result[symbol_dict[k]] = 1.0 * correct_count[k] / total_count[k]
        return {"accuracy": result, "wrong": wrong, "correct": correct_count, "total": total_count}

    def do_cross_validation_trained(self, feature_dicts, labels, secondary_labels, urls, clft):
        """
        Performs the cross validation for the chosen samples
        """
        vec = DictVectorizer(sparse=False)
        scaler = StandardScaler()
        print("\tcreating features array...")

        print("\tperforming stratified k-fold...")
        skf = StratifiedKFold(labels, n_folds=5)
        features_array = scaler.fit_transform(vec.fit_transform(feature_dicts))
        clf = RandomForestClassifier(n_estimators=14)

        correct_count = {}
        total_count = {}
        wrong = {}
        result = {}

        for train_index, test_index in skf:
            training_dicts = [features_array[t] for t in train_index]
            training_labels = [labels[t] for t in train_index]
            testing_points = [features_array[t] for t in test_index]
            testing_labels = [labels[t] for t in test_index]
            t = training_dicts + testing_points
            l = training_labels + testing_labels
            ind = list(train_index) + list(test_index)
            # clf.fit(training_dicts, training_labels)
            # clf.fit(testing_points, testing_labels)
            # clf.fit(t, l)
            for i in range(len(t)):
                label = l[i]
                for d in (correct_count, total_count):
                    if label not in list(d.keys()):
                        d[label] = 0
                total_count[label] += 1
                prediction = clft.predict(np.array(t[i]).reshape(1, -1))[0]
                if prediction == label:
                    correct_count[label] += 1
                else:
                    failed = False
                    folder = urls[ind[i]][0]
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
                            'feature_dict': feature_dicts[ind[i]],
                            'url': urls[ind[i]],
                            "predicted": symbol_dict[np.asscalar(prediction)],
                            "real": symbol_dict[label]
                        })

        for k in total_count:  # for each label
            result[symbol_dict[k]] = 1.0 * correct_count[k] / total_count[k]
        return {"accuracy": result, "wrong": wrong, "correct": correct_count, "total": total_count}

    def consolidate_accuracy_results_across_runs(self, run_results):
        """
        Gets all of the accuracy results from all of the runs
        """
        final_correct = 0
        final_total = 0
        final_accuracy_per_label = {}
        final_wrong = {}
        for run_result in run_results:
            accuracy = run_result['accuracy']  # TODO clean up using tuple probably
            wrong = run_result['wrong']
            correct_count = run_result['correct']
            total_count = run_result['total']
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
        print("average accuracy across all runs:")
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
                        [w['url'][0], w['real'], w['predicted'], w['url'][1], json.dumps(w['feature_dict'])]) + "\n")
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

    def print_chart_stats(self, types_lists):
        """
        Tells how many valid charts there are for each chart type
        """
        print("chart stats:")
        outp = {}
        for l in types_lists:
            outp[symbol_dict[l]] = len(types_lists[l])
        print(outp)

    def run_verification(self):
        """
        Runs the verification of the classifer given all of the presets
        """
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
        self.load_features()  # build the feature dicts, if they don't exist yet
        for collection in self.collection:
            self.organize_samples(collection, types_lists, secondary_labels)  # only call once
        # print secondary_labels

        if self.training:
            train_types_lists = {}
            for k in symbol_dict:
                train_types_lists[k] = []
            train_secondary_labels = {}
            for collection in self.training:
                self.organize_samples(collection, train_types_lists, train_secondary_labels)

            # make it so can only use samples in both
            for t in types_lists:
                if len(types_lists[t]) == 0:
                    train_types_lists[t] = []

            for t in train_types_lists:
                if len(train_types_lists[t]) == 0:
                    types_lists[t] = []

            # get features to train on all of training set
            t_feature_dicts, t_labels, t_urls = [], [], []
            self.select_subsets(t_feature_dicts, t_labels, t_urls, train_types_lists, repeat=False, training=True)

            t_feature_dicts, t_labels, t_urls = [], [], []
            self.select_subsets(t_feature_dicts, t_labels, t_urls, train_types_lists, repeat=False, training=True)

            vec = DictVectorizer(sparse=False)
            scaler = StandardScaler()
            skft = StratifiedKFold(t_labels, n_folds=5)
            features_arrayt = scaler.fit_transform(vec.fit_transform(t_feature_dicts))
            clft = RandomForestClassifier(n_estimators=14)
            clft.fit(features_arrayt, t_labels)

            total_time.append(time.time() - s)
            print("running experiment " + str(self.runs) + " time(s)...")

            # train on training set + test on test set
            for run_index in range(self.runs):
                print("executing run " + str(run_index) + "...")
                feature_dicts, labels, urls = [], [], []
                s = time.time()

                self.select_subsets(feature_dicts, labels, urls, types_lists, repeat=False)

                total_time.append(time.time() - s)

                result_dict = self.do_cross_validation_trained(feature_dicts, labels, secondary_labels, urls, clft)
                # result_dict = self.do_cross_validation(feature_dicts,labels,secondary_labels,urls)
                accuracy.append(result_dict)
        else:
            total_time.append(time.time() - s)
            print("running experiment " + str(self.runs) + " time(s)...")

            for run_index in range(self.runs):
                print("executing run " + str(run_index) + "...")
                feature_dicts, labels, urls = [], [], []

                s = time.time()
                if self.num_charts == -1:
                    self.select_subsets(feature_dicts, labels, urls, types_lists, repeat=False)
                else:
                    self.select_subsets(feature_dicts, labels, urls, types_lists)
                total_time.append(time.time() - s)
                result_dict = self.do_cross_validation(feature_dicts, labels, secondary_labels, urls)
                accuracy.append(result_dict)

        print("average extraction time:", np.sum(total_time) / self.runs)
        self.consolidate_accuracy_results_across_runs(accuracy)
        self.print_chart_stats(types_lists)


def main():
    """
    Runs the verification for some experiment on the classifier
    """
    parser = argparse.ArgumentParser(prog='Beagle Classifier')
    parser.add_argument('--train', help="Tells you the set that it is trained on")
    parser.add_argument('args', nargs="+", help="List of args for the classifier")

    sysargs = parser.parse_args()
    args = sysargs.args
    training = sysargs.train

    classifier = BeagleClassifier(args, training)
    classifier.run_verification()


if __name__ == '__main__':
    main()
