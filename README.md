# sportsbot

A tool for collecting Twitter conversations, fien-tuning and testing sentiment with GPT2 models from Huggingface's transformer library.

## Getting Started

To run this in google colab, you must first use colab-env to set up a vars.env file with Twitter API keys. The following steps are from [This tutorial](https://colab.research.google.com/github/apolitical/colab-env/blob/master/colab_env_testbed.ipynb#scrollTo=2rz2V-k1BZY9).

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

This function requires a search phrase, a list of words and/or phrases that should not appear in the conversation and a path to the file in which to store the `Conversation` objects. The default file is `output.jsonl`, which will be in the `sportsbot` folder by default. `Conversation` objects will contain each conversation in template form; you can either pass this into the `predict` function, or you can label the data for feature training.

## Labeling Data and Fine-tuning Models

To load the jsonl file of your collected data:

```sh
from sportsbot.datasets import read_data

#`validate_objs` will be a list of Conversation ojects.

validate_objs = read_data('data/multi_labeled_split_datasets/question_2_validate.jsonl')

```
If you want to create your datasets with a prompt different from the one specified when collecting the data, use `_prepare_conv_template` which will return a `Conversation` object with the new template in the `template` field.

```sh

from sportsbot.datasets importm_prepare_conv_template

_prepare_conv_template(conversation_obj, #this should be a list of `Tweet` objects, i.e. `conversation.thread`
                        topic,
                        question=None, # 1-5 integer specifying question number
                        end_prompt=None, # your custom prompt
                        conv_obj=Conversation #send the full `Conversation` object if you already have labels in `conv_obj`.label
)
```
To add labels to templates for `Conversation` objects for feature training, use `prepare_labeled_datasets`. If you want to keep the labels as integer values, set numeric to `True`.

```sh
from sportsbot.dataset import prepare_labeled_datasets

prepare_labeled_datasets(conversations, #list conversation objects
                        labels, #list of lables, ordered by the conversations objects list
                        jsonl_file='labeled_data.jsonl',
                        label_dict=None #make sure to send a label conversion dictionary
)
```

For fine-tuning with `Conversation` objects or foreign data:

```sh

from sportsbot.finetune import train

model = train(
    dataset, # either `Conversation` obects or templates
    question, # a string eg "Q1", for confusion matrix generation but can be easily customized
    validation_set=None, # not necessary if `eval_between_epochs` set to False
    validation_labels=None, # same as `validation_set`
    labels_dict=label_dict, #default is dict for all labels for all questions
    model=GPT2LMHeadModel, # can be any instantiated GPT2 model
    tokenizer=GPT2Tokenizer, # can be any instantiated GPT2 tokenizer
    batch_size=1, # can't go higher with Colab
    epochs=4, # used for gradient accumulaton because of batch size constraints on Colab
    lr=2e-5,
    max_seq_len=1024, # base this one model word embedding size
    warmup_steps=5000,
    gpt2_type="gpt2",
    device="cuda",
    output_dir=".",
    output_prefix="gpt2_fintune",
    save_model_on_epoch=True,
    eval_between_epochs=True,
    validation_file="validation",
    download=True, # if `model` parameter is an instantiated model, set to False else pre-trained model weights and tokenizer provided by Huggingface will be downloaded
    foreign_data=False, # True if dataset is not a list of `Conversation` objects
    plot_loss=True,
    prompt=None, # if dataset is not a list of `Conversation` objects, provide prompt for label masking
)

```

For models that have been feature trained or for zero-shot testing, use `predict`:

```sh
conversations = predict(test_convs, #a list of either conversations or templates
            tokenizer,
            model,
            device="cuda",
            num_top_softmax=20, # return top-k
            json_file_out='add_stats_output.jsonl',
            labels=None, # labels for `test_convs`, ordered with respect to `test_convs`
            labels_dict=None,
            foreign_data=False, # false is test_convs are `Conversation` objects
            logit_labels_only=False #probability taken only for classification labels
        )
```
Visualization functions such as `create_confusion_matrix` can be found in `sportsbot.finetune`, and can be used as stand alone functions on saved validation data:

```sh
from sportsbot.finetune import create_confusion_matrix
#labels_dict_neutral is the labels conversion dictionary for this dataset
#`conversations`` is a list of validation data returned by `predict`

for count in range(len(conversations)):
    for stats in conversations: 
        ground_truth = [labels_dict_neutral["all_values"][stats[str(index)][2]] for index in range(len(stats)-5)]
        model_predictions = [stats[str(index)][3][0][0] for index in range(len(stats)-5)]
        create_confusion_matrix(
                                ground_truth,
                                model_predictions,
                                "Q2",
                                epoch=count,
                                lr=2e-5,
                                output_prefix="model_detals",
                                out_file="output_file_name"
        )
```
