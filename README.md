# vis-type-predictor

This code predicts visualization type using SVG/images.
The annotator code from the Beagle project: http://www.cs.umd.edu/~leilani/static/papers/CHI2018_cr_battle_02_05_2018.pdf

The decision tree classifier performs stratified 5-fold cross validation on the
dataset and prints the accuracy per chart type (per run and averaged over all runs) and the overall accuracy.

Installation requirements: numpy, scikit-learn

**Instructions to run:**

To run, type `python decisiontree.py <data directory> <num times to repeat experiment> <num of charts to sample per vis type> <use text>`

`<data directory>` (required) is the path to the directory storing chart data.

It must have the following format:  
- A **"urls.txt"** file directly under `<data directory>`, where each row is space-separated and contains the chart id,
url, integer indicating chart type
  If multiple chart types, separate the types with *","*, and an optional *"i"* flag where if present, tells the classifier to skip the chart.  
- There must also be two directories **"charts"** and **"images"**
directly under `<data directory>`, where "charts" contains a folder for each chart labeled with the chart id, and each chart folder stores the svg of the chart titled **"svg.txt"**.  
- **"images"** contains a png snapshot of each chart titled **"<chart id>.png"**
- `<num times to repeat experiment>` *(optional, default = 1)*. The number of times to run the experiment.
- `<num of charts to sample per vis type>` *(optional, default = 10)*. The number of chart samples to select (randomly) per chart type, to ensure a fair
representation from each chart type.
  If there are not enough samples for a chart type, the same sample may be selected multiple times.
- `<use text>` *(optional, default = True) "true/false"* indicating whether to use chart text features.
- The dictionary **"symbol_dict"** in `decisiontree.py` must be filled mapping each integer to the chart type it represents.
