# Training Data

Each zipped folder will contain a day's worth of datasets collected. The number of conversations in each file will vary anywhere between 0 and 50 (20 for Oct. 9th).

After unzipping, to read in all of the data in a folder:

```sh
import os
from sportsbot.datasets import read_data

dir_path = 'unlabeled_training_data/{unzipped_file_name}/'
file_names = os.listdir(dir_path)
data = []
for name in file_names:
    data += read_data(dir_path+name)
```

Each `Conversation` object in the `data` list will have a training template ready to be label. 

```sh
for conv in data:
    print(conv.template)
```

For now, to add labels to the template, make a list (`labels`) of the labels for each `Conversation` object in `data` (making sure to mirror the ordering of the labels with respect to the objects in `data`) and call the function `prepare_labeled_datasets` to add your labels to the training template:

```sh
from sportsbot.datasets import prepare_labeled_datasets

labeled_data = prepare_labeled_datasets(data, labels, jsonl_file=labeled_data_{collection date})
```

The function will return the labeled objects and save them in another jsonl file. Please use the naming convention `labeled_data_{collection date}`, where collection date is the date in the data file's name.