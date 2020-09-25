# sportsbot

A tool for streaming Twitter arguments/discussions and using a few-shot-trained GPT-2 model to classify which side of the argument a (for now) randomly chosen participant is on.

## run

To run this in google colab, you must first use colab-env to set up a vars.env file with Twitter API keys. The following steps are from [This tutorial] (https://colab.research.google.com/github/apolitical/colab-env/blob/master/colab_env_testbed.ipynb#scrollTo=2rz2V-k1BZY9) will take you through setting it up.

    1. To install, run 
    ```sh
    !pip install colab-env --upgrade
    ``` 

    2. Import:

    ```sh
    import colab_env
    ``` 

    Importing the module will set everything up; it create vars.env if it doesn't already exist and if it does, it will load your environment variables. If your drive isn't alreay mounted, it will walk you through authenticating your colab session, and your account's google drive should be mounted.

    3. To add or change the keys run: 

    ```sh
    colab_env.envvar_handler.add_env("KEY", "value", overwrite=True)
    ```

   Make sure to name them: "AKEY" (API Key), "ASECRETKEY" (API Secret Key), "ATOKEN" (access token), "ASECRET" (secret access token), use:
     

If for some reason your google drive hasn't been mounted, run:

```sh
from google.colab import drive
drive.mount(‘/content/gdrive’)
```

mkdir a project folder, then cd into that folder and run:

```sh
! git clone https://github.com/credwood/SportsBot.git
```

cd into SportsBot and install the dependencies by running:

```sh
!pip install -r requirements.txt
```
The environment in which this module was deveoped was a pyenv virtualenv 3.7.6.

Once the dependencies are installed, cd into sportsbot and you can get started by:

```sh
from sportsbot.conversations import get_conversations

data = get_conversations(
            "my search phrase", 
            ['"phrases to exclude"', 'bad', 'word'], jsonlines_file='my_output.jsonl'
            )
```

This function requires a search phrase and a list of words or phrases that should not appear in the conversation* and a file path to save jsonlines file containing the conversations. The default file will be called output.jsonl and will be created in your project folder.

To test the classifier, you will need to create a list of labels for the conversations, and then you can import and run `few_shot_train`:

```sh
from sportsbot.inference import few_shot_train
training_data = few_shot_train(
                    data, 
                    my_labels, 
                    topic, 
                    jsonline_file='my_output.jsonl')
```

the function will return and add the statistics for each conversation will be added to each `Conversation` dataclass.

*for now the filter if only applied to the initial tweet found in the conversation.
