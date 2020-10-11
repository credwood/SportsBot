"""
Functions for testing Hugginface's GPT-2 trained model
"""
from collections import defaultdict
#import tensorflow as tf
from scipy.special import softmax
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .datasets import _prepare_few_shot_testing_set, _save_data, ConversationPrompt

def download_model_tokenizer(model_type="gpt2"):
    """
    colab can run GPT2 mdoels: 'gpt2', 'gpt2-medium, 'gpt2-large'
    """
    tokenizer_instantiate = GPT2Tokenizer.from_pretrained(model_type)
    tokenizer_instantiate.pad_token = tokenizer_instantiate.eos_token

    # add the EOS token as PAD token to avoid warnings
    model_instantiate = TFGPT2LMHeadModel.from_pretrained(model_type,
                                            pad_token_id=tokenizer_instantiate.eos_token_id,
                                            return_dict=True
                                            )
    return model_instantiate, tokenizer_instantiate

def few_shot_test(test_data,
                    topic,
                    training_conversations,
                    training_labels,
                    tokenizer,
                    model,
                    num_top_softmax=15,
                    test_labels=False,
                    jsonlines_file_out='add_stats_output.jsonl'
    ):
    """
    Function for experimentation. Takes a few records for training and their labels,
    a test set and --if desired--its labels, the conversation topic on which to classify,
    a jsonlines file path for the output (SoftMax values for each conversation along with
    the conversation in template form), a GPT-2 model and corresponding tokenizer.
    For labeled test data, the function will return the accuracy, dictionary of 15 tokens with
    highest softmax values and a list of tuples with the model's answer and the
    correct label. For unlabeled test data, the function will return a dictionary
    of the top SoftMax values for each conversation.
    """
    if test_labels:
        if len(test_data) != len(test_labels):
            raise AssertionError ("Must have an equal number of test cases and test labels")
    if len(training_conversations) != len(training_labels):
        raise AssertionError ("Must have an equal number of trianing cases and training labels")
    #probabilities_dict = defaultdict()
    model_answers = []
    templated_conversations = []
    confidence = defaultdict()
    shots_and_tests, prompts = _prepare_few_shot_testing_set(
                                                            training_conversations,
                                                            test_data,
                                                            topic,
                                                            training_labels
                               )
    for i, tweets in enumerate(shots_and_tests):
        input_ids = tokenizer.encode(tweets,return_tensors='tf')
        output = model(input_ids)
        predicted_prob = softmax(output.logits[0, -1, :], axis=0)
        #probabilities_dict[f"{i+1}_test"] = predicted_prob

        top_softmax = _top_softmax(predicted_prob,tokenizer,num_top_softmax)
        model_answers.append(list(top_softmax[0].keys())[0])
        confidence[i] = top_softmax
        templated_conversations.append(ConversationPrompt(test_data[i].template, prompts, top_softmax))
    _save_data(templated_conversations,jsonlines_file_out)

    if test_labels:
        accuracy = _calculate_accuracy(test_labels, model_answers)
        statistics = {
                        "accuracy": accuracy,
                        "confidence_dict": confidence,
                        "model_answeres_vs_labels": list(zip(model_answers, test_labels))
                        }

        return statistics
    else:
        return confidence

def predict(test_convs,
            tokenizer,
            model,
            num_top_softmax=15,
            jsonlines_file_out='add_stats_output.jsonl'
        ):
    """
    Function for finetuned model. saves and returns `Conversation` objects
    with model statistics (top SoftMax values for each conversation).
    """
    conversations = []
    for i, test_conv in enumerate(test_convs):
        tweet_template = test_conv.template
        input_ids = tokenizer.encode(tweet_template,return_tensors='tf')
        output = model(input_ids)
        predicted_prob = softmax(output.logits[0, -1, :], axis=0)
        top_softmax = _top_softmax(predicted_prob,tokenizer, num_top_softmax)
        test_convs[i].model_statistics = top_softmax
        conversations.append(test_convs[i])
    _save_data(conversations,jsonlines_file_out)
    return conversations

def _calculate_accuracy(labels, model_answers):
    correct = 0.
    for i, answer in enumerate(labels):
        if  model_answers[i] == answer:
            correct += 1
    return correct/len(labels)

def _top_softmax(prob_dict, tokenizer, num_tokens):
    num_tokens = min(len(prob_dict), num_tokens)
    sorted_indices = np.argsort(prob_dict)[::-1][:num_tokens]
    return [{tokenizer.decode([index]): str(prob_dict[index])} for index in sorted_indices]
