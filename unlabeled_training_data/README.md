Each zipped folder will contain a day's worth of datasets collected. Because of the rate limit, the number of conversations in each file can vary anywhere between 1 and 50 (20 for the first day).

After unzipping, to read in the data:

```sh
import os
from sportsbot.datasets import read_data

file_names = os.listdir('unlabeled_training_data/{unzipped_file_name}/')
data = []
for name in file_names:
    data += read_data(name)
```

Each `Conversation` object in the `data` list will have a training template ready to be label. 

```sh
for conv in data:
    print(conv.template)
```

For now, to add labels to the template, make a list (`labels`) of the labels for each object in `data` (making sure to mirror the ordering of the labels with respect to the objects in `data`) and call the function `prepare_labeled_datasets`:

```sh
from sportsbot.datasets import prepare_labeled_datasets

labeled_data = prepare_labeled_datasets(data, labels, jsonl_file=labeled_data_{collection date})
```

The function will return the labeled objects and save them in another jsonl file. Please adhere to the file naming convention `labeled_data_{collection date}`, where collection date is the date in the data file's name.