# sportsbot

A tool for streaming Twitter arguments/discussions and using a few-shot-trained GPT-2 model to classify which side of the argument a randomly chosen participant (randomly chosen for now) is on.

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

This function requires a search phrase, a list of words and/or phrases that should not appear in the conversation and a path to the file in which to store the `Conversation` objects. The default file is `output.jsonl`, which will be in the `sportsbot` folder. `Conversation` objects will contain each conversation in template form; you can either pass this into the `predict` function, or you can label the data for feature training.

## Labeling Data and Training Models

To add labels to templates for `Conversation` objects for feature training, use `prepare_labeled_datasets`. If you want to keep the labels as integer values, set numeric to `True`.

```sh
from sportsbot.dataset import prepare_labeled_datasets

prepare_labeled_datasets(conversations, labels, jsonl_file='labeled_data.jsonl', numeric=False)
```

For few-shot training, you can create a list of labels for the testing sets and run `few_shot_test`:

```sh
from sportsbot.inference import few_shot_train, download_model_tokenizer

model, tokenizer = download_model_tokenizer()
training_data = few_shot_test(test_data,
                    topic,
                    training_conversations,
                    training_labels,
                    tokenizer,
                    model,
                    num_top_softmax=15,
                    test_labels=False,
                    jsonlines_file_out='add_stats_output.jsonl'
    ):
```

For each conversation, the function will write a `ConversationPrompt` object to the `jsonlines_file_out` file. Each object will contain the conversation in template-form (without the training conversations) and a list of the 15 tokens with the largest SoftMax values. The function will return the batch accuracy, 15 largest SoftMax values for each conversation and a list of (model predictions vs labels).

For models that have been feature trained, use `predict`:

```sh
conversations = predict(test_convs,
            tokenizer,
            model,
            num_top_softmax=15,
            jsonlines_file_out='add_stats_output.jsonl'
        ):
```
