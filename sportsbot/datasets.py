# pylint: disable=E1101
"""
Tools for writing to jsonlines files and processing
`Conversation` and `Tweet` objects.
"""
import random
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

@dataclass_json
@dataclass
class Conversation:
    """
    Each `Conversation` object
    will hold a list of `Tweet` objects,
    making up a conversation thread. If the conversation
    has been run through the model, it will also hold a
    a dictionary of statistics
    """
    thread: dict
    model_statistics: dict

def _save_data(data, file):
    """
    writes `Conversation` objects to a jsonlines file
    """
    for conversation in data:
        with jsonlines.open(file, mode='w') as writer:
            writer.write(conversation.to_json())


def _prepare_testing_set(shots, data_to_test,topic,labels):
    templated_prompts = _few_shot_template(shots, topic, labels)
    test_convs = []
    for conv in data_to_test:
        names = set([])
        conversation_str = ''
        for tweet in conv.thread:
            conversation_str += tweet.content + '\n'
            names.add(tweet.user_handle)
        name = random.choice(list(names))
        end_prompt = f"\n--\nQuestion: Does {name} like {topic}? \nAnswer:"
        test_convs.append(templated_prompts+ conversation_str + end_prompt)
    return test_convs

def _few_shot_template(shots, topic, labels):
    accumulate_prompts = ''
    for index, shot in enumerate(shots):
        conversation_thread = shot.thread
        names = set([])
        conversation_str = ''
        for tweet in conversation_thread:
            conversation_str += tweet.content + '\n'
            names.add(tweet.user_handle)
        name = random.choice(list(names))
        end_prompt = f"\n--\nQuestion: Does {name} like {topic}?\nAnswer: {labels[index]}"
        conversation_str = conversation_str + end_prompt
        accumulate_prompts += conversation_str +"\n\nnext dialogue\n"
    return accumulate_prompts

def _add_stats(conversation,stats,json_file):
    with jsonlines.open(json_file) as reader, jsonlines.open(json_file, mode='w') as writer:
        for obj in reader:
            obj = obj.from_json(obj)
            if obj==conversation:
                obj.model_statistics = stats
            writer.write(obj.to_json())
