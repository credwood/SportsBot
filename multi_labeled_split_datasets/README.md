# Fine-tuning Data

To read in all of the data in a folder:

```sh
import os
from sportsbot.datasets import read_data

dir_path = 'multi_labeled_split_datasets/'

data = read_data(dir_path+"specific_file_name")
```

Each `Conversation` object in the `data` list will have a fine-tuning template along with a list of labels. If the template is for question 1, conv.label[0] is the correct label and conv.label[-1] is the topic about which the quetsion is answered, conv.handle_tested is the person in the conversation about which the question is answered.

If it's fine-tuning data (...train.jsonl), the natural language label will the last token of the template (see the main README for the label dictionary used for converting numeric labels to natural language).

```sh
for conv in data:
    print(conv.template)
    print(conv.label)
```
