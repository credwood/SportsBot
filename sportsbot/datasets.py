# pylint: disable=E1101
"""
Tools for writing to jsonlines files and processing
`Conversation` and `Tweet` objects.
"""
import random
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import jsonlines

@dataclass_json
@dataclass
class Tweet:
    """
    `Tweet` object created for each
    tweet in a conversation.
    """
    user_id: int
    user_handle: str
    display_name: str
    content: str
    language: str
    date_time: str
    num_followers: int
    num_followed: int
    profile_description: str

@dataclass_json
@dataclass
class Conversation:
    """
    Each `Conversation` object
    will hold a list of `Tweet` objects,
    making up a conversation thread.
    """
    thread: list
    label: str
    template: str
    model_statistics=list

@dataclass_json
@dataclass
class ConversationPrompt:
    """
    Dataclass for holding the templated conversation
    and the labels, or an empyt list if the dataset is not
    labeled.
    """
    test_text: str
    prompt_text: str
    model_statistics: list

@dataclass_json
@dataclass
class Account:
    """
    dataclass to store info about study participants
    """
    account_id: int
    account_name: str
    bot: str
    sentiment: dict
    engagement: dict


def _save_data(data, file):
    """
    writes `Conversation` objects to a jsonlines file
    """
    with jsonlines.open(file, mode='w') as writer:
        for conversation in data:
            writer.write(conversation.to_json())

def read_data(file,conversation_obj=True):
    """
    reads jasonl data into either a `Conversation`
    object, or a `ConversationPrompt` object.
    """
    conversations = []
    with jsonlines.open(file) as reader:
        if conversation_obj:
            for conv in reader:
                conv = json.loads(conv)
                conv_list = []
                for tweet in conv["thread"]:
                    conv_list.append(Tweet(
                                            int(tweet["user_id"]),
                                            tweet["user_handle"],
                                            tweet["display_name"],
                                            tweet["content"],
                                            tweet["language"],
                                            tweet["date_time"],
                                            int(tweet["num_followers"]),
                                            int(tweet["num_followed"]),
                                            tweet["profile_description"]
                                           )
                                    )
                conversations.append(
                                Conversation(
                                            conv_list,
                                            conv["label"],
                                            conv["template"],
                                            conv["model_statistics"]
                                            )
                                    )
        else:
            for conv in reader:
                conv = json.loads(conv)
                conversations.append(
                                ConversationPrompt(
                                                    conv["test_text"],
                                                    conv["prompt_text"],
                                                    conv["model_statistics"]
                                                  )
                                    )

    return conversations

def _prepare_conv_template(conversation, topic):
    conversation_str = ''
    new_line = '\n'
    names = set([])
    for tweet in conversation:
        conversation_str += f"{tweet.user_handle}: {tweet.content}{new_line}"
        names.add(tweet.user_handle)
    name = random.choice(list(names))
    end_prompt = (f"{new_line}--{new_line}"
                    f"Question: Does {name} like {topic}? {new_line}Answer:")
    return Conversation(conversation, '', conversation_str + end_prompt)

def prepare_labeled_datasets(conversations, labels, jsonl_file='labeled_data.jsonl'):
    """
    add template to `Conversation` object. Returns and saves objects.
    """
    if len(conversations) != len(labels):
        raise AssertionError("Must have an equal number of conversations and labels")
    for index, _ in enumerate(conversations):
        conversations[index].template = conversations[index].template + labels[index]
        conversations[index].label = labels[index]
    _save_data(conversations,jsonl_file)
    return conversations

def _prepare_few_shot_testing_set(shots, data_to_test,topic,few_shot_labels):
    templated_prompts = _few_shot_template(shots, topic, few_shot_labels)
    test_convs = _few_shot_template(data_to_test,
                                        topic,
                                        few_shot_labels,
                                        templated_prompts=templated_prompts,
                                        test_data=True
                                        )
    return test_convs

def _few_shot_template(shots, topic, few_shot_labels, templated_prompts=None, test_data=False):
    accumulate_prompts = ''
    test_convs = []
    test_template = []
    new_line = '\n'
    for index, shot in enumerate(shots):
        conversation_thread = shot.thread
        names = set([])
        conversation_str = ''
        for tweet in conversation_thread:
            conversation_str += f"{tweet.user_handle}: {tweet.content}{new_line}"
            names.add(tweet.user_handle)
        name = random.choice(list(names))
        if test_data:
            end_prompt = (f"{new_line}--{new_line}"
                            f"Question: Does {name} like {topic}? {new_line}Answer:")
            conv_prompt = conversation_str + end_prompt
            test_template.append(conv_prompt)
            test_convs.append(templated_prompts + conv_prompt)
        else:
            end_prompt = (f"{new_line}--{new_line}Question: Does {name} like {topic}?{new_line}"
                                                f"Answer: {few_shot_labels[index]}"
                                                f"{new_line}{new_line}next dialogue{new_line}")
            accumulate_prompts += (conversation_str
                                    + end_prompt)
    return (test_convs, test_template, templated_prompts) if test_data else accumulate_prompts
