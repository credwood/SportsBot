# sportsbot

A collection of functions for collecting Twitter conversations, fine-tuning and testing sentiment with GPT2 models from Huggingface's transformer library.

NOTE: Twitter recently upgraded their API but there hasn't been a Tweepy release that addresses the changes. When there is, I will upgrade this code.

## Getting Started

If you want to use google colab to connect to Twitter's API to collect conversations, you must first use colab-env to set up a vars.env file with Twitter API keys. The following steps are from [This tutorial](https://colab.research.google.com/github/apolitical/colab-env/blob/master/colab_env_testbed.ipynb#scrollTo=2rz2V-k1BZY9).

1. To install, run:

    ```sh
    ! pip install colab-env --upgrade
    ```

2. Import:

    ```sh
    import colab_env
     ```

Importing the module will set everything up; it create vars.env if it doesn't already exist and if it does, it will load your environment variables. It will walk you through authenticating your colab session, after which your account's drive should be mounted. If you want to work in a directory on your drive, run and cd into:

```sh
from google.colab import drive
drive.mount(‘/content/gdrive’)
```

3. To add or change the API keys, run:

```sh
colab_env.envvar_handler.add_env("KEY", "value", overwrite=True)
```

The module requires that you name them: "AKEY" (API Key), "ASECRETKEY" (API Secret Key), "ATOKEN" (access token), "ASECRET" (secret access token)

Clone:

```sh
! git clone https://github.com/credwood/SportsBot.git
```

cd into SportsBot and install the dependencies:

```sh
! pip install -r requirements.txt
```
(The environment in which this module was deveoped was a pyenv virtualenv 3.7.6.)

## Collecting Conversations

Once the dependencies are installed, you can get started with:

```sh
from sportsbot.conversations import get_conversations

data = get_conversations(search_terms,
                        filter_terms,
                        template_topic,
                        jsonlines_file='output.jsonl',
                        max_conversation_length=10):
```

This function returns a list of `Conversation` objects. It requires a search phrase (`search_terms`), a list of words and/or phrases that should not appear in the conversation (`template_topc`), the topic that should be used for the template (`template_topic`) and a path to the file in which to store the `Conversation` objects. The default file is `output.jsonl`, which will be in the `sportsbot` folder by default. `Conversation` objects will contain each conversation in template form; you can either pass this into the `predict` function, or you can label the data for feature training.

## Labeling Data and Fine-tuning Models

To load jsonl `Conversation` files:

```sh
from sportsbot.datasets import read_data

#`validate_objs` will be a list of `Conversation` ojects with
# templates for validating models fine-tuned for Question 2.

validate_objs = read_data('data/multi_labeled_split_datasets/question_2_validate.jsonl')

```
The end-prompt used for the default template (generated when conversations are collected) is `f"{new_line}--{new_line}Question: Does {name} like {topic}? {new_line}Answer:"`. If you want to create your own prompt, you can write your own function; `_prepare_conv_template` function in `sportsbot.datasets` might be a useful starting point.

To add labels to `Conversation` objects' templates for feature training, you can use `prepare_labeled_datasets`, or write your own simple function if the specifics of this one don't work for you. This function will return (and save) a list of the labeled `Conversation` objects.

```sh
from sportsbot.dataset import prepare_labeled_datasets

labeled_conversations = prepare_labeled_datasets(conversations, #list of conversation objects
                        labels, #list of lables, ordered by the conversations objects list
                        jsonl_file='labeled_data.jsonl',
                        label_dict=None #make sure to send a label conversion dictionary, even if it's just an identity map (see below for label dictionary formatting).
)
```

For fine-tuning with `Conversation` objects or foreign data use `train` from sportsbot.finetune. The function will return the fine-tuned model. You have the option to save validation statistics, graphs and check-pointed weights:

```sh

from sportsbot.finetune import train

model = train(
    dataset, # either `Conversation` obects or templates
    question, # a string eg "Q1". Used for confusion matrix generation but can be easily customized.
    validation_set=None, # not necessary if `eval_between_epochs` set to False
    validation_labels=None, # not necessary if `eval_between_epochs` set to False
    labels_dict=label_dict, # default is dict for all labels for all five questions, but you should make your own (see below)
    model=GPT2LMHeadModel, # can be any instantiated GPT2 model
    tokenizer=GPT2Tokenizer, # can be any instantiated GPT2 tokenizer
    batch_size=5, # this is used for gradient accumulation, batch size is always 1 because of Colab GPU memory limitations
    epochs=4,
    lr=2e-5, #learning rate
    max_seq_len=1024, # base this on size of model's word embedding
    warmup_steps=5000, # scheduler warm up steps
    gpt2_type="gpt2", # specify which GPT-2 model
    device="cuda",
    output_dir=".", # directory in which to save checkpointed model weights
    output_prefix="gpt2_fintune", # set file name of checkpointed model weights
    save_model_on_epoch=True, # True if you want to save checkpointed weights after each epoch
    eval_between_epochs=True, # if True, will save a json file of validation statistics after each epoch
    validation_file="validation", # name of validation file
    download=True, # if `model` parameter is an instantiated model, set to False else pre-trained model weights and tokenizer provided by Huggingface will be downloaded
    foreign_data=False, # True if dataset is not a list of `Conversation` objects
    plot_loss=True, # will plot loss and accuracy for validation and fine-tuning datasets for each epoch, will save the figure as `f"loss_accuracy_graph_{output_prefix}.png"`
    prompt=None, # if dataset is not a list of `Conversation` objects, must provide the prompt used in order to mask label tokens
)

```

For models that have been feature trained or for zero-shot testing, use `predict`. This funcion will return (and save) a large dictionary of validation statistics for each conversation, as well as statistics for the entire dataset:

```sh
conversations = predict(test_convs, #a list of either conversations or templates
            tokenizer, # instantiated tokenizer
            model, # instantiated model
            device="cuda",
            num_top_softmax=20, # will save top-20
            json_file_out='add_stats_output.jsonl',
            labels=None, # labels for `test_convs`, ordered with respect to `test_convs`
            labels_dict=None, # add your label conversion dictionary. see example below.
            foreign_data=False, # false is test_convs are `Conversation` objects
            logit_labels_only=False #probability taken only for classification labels
        )
```

In the returned (or saved) validation stats dictionary, `conversations`, for the ith conversation the dictionary contains: a list containing the template tested, softmax values for all labels, the ground truth value, the top 20 (default) softmax values

To access data for the ith conversation:
```sh
conversations[str(i)] = [tweet_template, all_label_softmax, label, top_softmax]
```
for the entire dataset:

```sh
conversations["accuracy"] # accuracy of input dataset
conversations["soft_accuracy"] # soft accuracy of input dataset
conversations["validation_loss"] # loss for predicted token only
conversations["hist_data"] = [Counter(labels), Counter(answers)] # count of ground truth and predicted values
conversations["label_softmaxes"] # dictionary of average softmax values for each of the class labels. Will probably refactor out.
```

Visualization functions such as `create_confusion_matrix` can be found in `sportsbot.finetune`, and can be used as stand alone functions on validation data:

```sh
from sportsbot.finetune import create_confusion_matrix
#labels_dict_neutral is the labels conversion dictionary for this dataset (see example below)
#`conversations_list`` is a list of validation data returned from multiple runs of `predict`
#"Q2" is used to identify which labels to use for the matrix.
#You can customize this by specifying your own `classes` dictionary. See the source code for how to structure it.

for count, stats in enumerate(conversations_list): 
    ground_truth = [labels_dict_neutral["all_values"][stats[str(index)][2]] for index in range(len(stats)-5)]
    model_predictions = [stats[str(index)][3][0][0] for index in range(len(stats)-5)]
    create_confusion_matrix(
                            ground_truth,
                            model_predictions,
                            "Q2",
                            epoch=count,
                            lr=2e-5,
                            output_prefix="model_detals",
                            classes = classes_dict,
                            out_file="output_file_name"
    )
```

Label dictionary example:

If you want to use your own label conversion dictionary, follow the same format and include the same three sub-dictionaries, even if some have dummy or identity values. `"bucketed_values"` is used to calculate the soft accuracy and the `"baseline_accuracy"` value tracks the maximum accuracy the validation dataste would reach if the model converges to the dominant label in the fine-tuning dataset.

```sh
label_dict = {"all_values": {
                  1: " No",
                  2: " Remote",
                  3: " Unsure",
                  4: " Probably",
                  5: " Yes",
                  6: " Neutral",
                  7: " None",
                  8: " Positive",
                  9: " Defensive",
                  10: " Negative",
                  11: " Opposition",
                  12: " Discussion",
                  13: " Agreement"
              },
              "bucketed_labels":{
                  1: [" No", " Remote"],
                  2: [" No", " Remote"],
                  3: [" Unsure"],
                  4: [" Probably", " Yes"],
                  5: [" Probably"," Yes"] ,
                  6: [" Neutral", " None"],
                  7: [" None", " Neutral"],
                  8: [" Positive"],
                  9: [" Defensive"],
                  10: [" Negative"],
                  11: [" Opposition",],
                  12: [" Discussion"],
                  13: [" Agreement"]    
              },
              "baseline_accuracy": 0.333
}
```
