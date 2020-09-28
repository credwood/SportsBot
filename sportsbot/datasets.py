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
    a list of the 15 tokens with highest SoftMax values
    """
    thread: list
    model_statistics: list

@dataclass_json
@dataclass
class ConversationPrompt:
    """
    Dataclass for holding the templated conversation
    and its model statistics
    """
    text: list
    model_statistics: list

def _save_data(data, file):
    """
    writes `Conversation` objects to a jsonlines file
    """
    with jsonlines.open(file, mode='w') as writer:
        for conversation in data:
            writer.write(conversation.to_json())

def _prepare_testing_set(shots, data_to_test,topic,few_shot_labels):
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
    for index, shot in enumerate(shots):
        conversation_thread = shot.thread
        names = set([])
        conversation_str = ''
        for tweet in conversation_thread:
            conversation_str += f"{tweet.user_handle}: "+tweet.content + '\n'
            names.add(tweet.user_handle)
        name = random.choice(list(names))
        if test_data:
            end_prompt = f"\n--\nQuestion: Does {name} like {topic}? \nAnswer: "
            conv_prompt = conversation_str + end_prompt
            test_template.append(conv_prompt)
            test_convs.append(templated_prompts + conv_prompt)
        else:
            end_prompt = f"\n--\nQuestion: Does {name} like {topic}?\nAnswer: {few_shot_labels[index]}"
            conversation_str = conversation_str + end_prompt
            accumulate_prompts += conversation_str +"\n\nnext dialogue\n"
    return (test_convs, conv_prompt) if test_data else accumulate_prompts
