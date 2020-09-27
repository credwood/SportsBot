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

Once the dependencies are installed, you can get started with:

```sh
from sportsbot.conversations import get_conversations

data = get_conversations(
            "my search phrase", 
            ['"phrases to exclude"', 'bad', 'word'], jsonlines_file='my_output.jsonl'
            )
```

This function requires a search phrase, a list of words and/or phrases that should not appear in the conversation* and a path to the file in which to store the `Conversation` objects. The default file is `output.jsonl`, which will be in the `sportsbot` folder.

To test the classifier, you will need to create a list of labels for the training and testing sets. Import and run `few_shot_train`:

```sh
from sportsbot.inference import few_shot_train

training_data = few_shot_train(test_data,
                    test_labels,
                    topic,
                    training_conversations,
                    few_shot_labels,
                    jsonlines_file_out='add_stats_output.jsonl'
                    )
```

The function will return the tokens and SoftMax values of the 15 most likely answers, and it will add them to each `Conversation` object which will be saved in a new output file. The function will also return the batch accuracy, SoftMax values and a list of (model predictions vs labels).

*For now the filter is only applied to the initial tweet found in the conversation.
