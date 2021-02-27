# Fine-tuning Data

To read in the data of a specific file:

```sh
from sportsbot.datasets import read_data

dir_path = 'multi_labeled_split_datasets/'

data = read_data(dir_path+"specific_file_name")
```

Each `Conversation` object in the `data` list will have a fine-tuning template along with a list of labels. If the template is for question 1, conv.label[0] is the correct label; for question 2, conv.label[1] and so on. conv.label[-1] is the topic about which the quetsion is answered, conv.handle_tested is the person in the conversation about which the question is answered.

If it's fine-tuning data (...train.jsonl), the natural language label will be the last token of the template (see the main README for the label dictionary used for converting numeric labels to natural language, or the questions below).

```sh
for conv in data:
    print(conv.template)
    print(conv.label)
```
The labels and their natural language mappings for the five questions are below. Parentheticals contain how responses are hard coded and stored in the `label` field in each `Conversation` object. All responses are encoded as single tokens.

For more information on the labeling methadology, see [this document](https://docs.google.com/document/d/1UYfLlOY5_peg9ZGkIYHlY-fb7qqenyE8ml8NaiiaX1E/edit)

```sh
Question 1, Affinity for subject:
    Does {name} like {topic}? 
        (1) No
        (2) Remote (can’t find a better single token synonym)
        (3) Unsure
        (4) Probably
        (5) Yes
        (6) if neutral entity e.g. national media outlet/media outlet not explicitly pro {subject} or the leagues themselves, e.g. NFL, NBA: Neutral 
        (7) if not relevant/about fantasy or gambling/esports/not in english: None
Question 2, Sentiment:
    What is {name}’s sentiment about {topic} in this conversation? 
    (pared down questions: Is {name}’s sentiment about {topic} positive, negative or neutral?)
        (8) Positive
        (9) Defensive
        (6) if no bias detected or not directly addressed: Neutral
        (10) Negative
        (3) if unknown: Unsure 
        (7) if not relevant/not in English: None
Question 3, Tone:
    In this conversation, with respect to the person/people {name} is responding to, is {name} in:
        (11) Conversations that have gotten personal: Opposition
        (12) Discussions, even heated ones: Discussion 
        (13) Strong agreement (echoing) of a point: Agreement
        (3) Unsure
        (7) If {name} is the only person in the conversation or if they are the first tweet with no subsequent tweet: None
Question 4, Sarcasm:
    Does {name} say anything sarcastically?
        (5) Yes 
        (1) No
        (3) Unsure
        (7) if not in english: None
Question 5, Trolling:
    Is {name} trolling?
        (5) Yes
        (1) No
        (3) Unsure
        (7) if not in english: None
```