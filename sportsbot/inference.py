"""
Functions for testing Hugginface's GPT-2 trained model
"""
from collections import defaultdict
#import tensorflow as tf
from scipy.special import softmax
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .datasets import _prepare_testing_set, _add_stats

tokenizer_instantiate = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer_instantiate.pad_token = tokenizer_instantiate.eos_token

# add the EOS token as PAD token to avoid warnings
model_instantiate = TFGPT2LMHeadModel.from_pretrained("gpt2-large",
                                        pad_token_id=tokenizer_instantiate .eos_token_id,
                                        return_dict=True
                                        )

def few_shot_train(data,
                    labels,
                    topic,
                    training_conversations,
                    few_shot_labels,
                    jsonlines_file='output.jsonl',
                    tokenizer=tokenizer_instantiate ,
                    model=model_instantiate
                    ):
    """
    Function for experimentation. Takes a few labeled records for training,
    a dataset and its labels, the conversation topic on which to classify,
    a jsonlines file path containing the data objects and updates the file with the
    model's prediction and returns the accuracy, dictionary of 15 tokens with
    highest softmax values and a list of tuples with the model's answer and the
    correct label.
    """
    #probabilities_dict = defaultdict()
    model_answers = []
    confidence = defaultdict()
    conversations = _prepare_testing_set(training_conversations, data, topic, few_shot_labels)
    for i, tweets in enumerate(conversations):
        input_ids = tokenizer.encode(tweets,return_tensors='tf')
        output = model(input_ids)
        predicted_prob = softmax(output.logits[0, -1, :], axis=0)
        #probabilities_dict[f"{i+1}_test"] = predicted_prob

        max_word=tokenizer.decode(np.where(predicted_prob==max(predicted_prob)))
        model_answers.append(max_word)
        top_softmax = _top_softmax(predicted_prob,tokenizer)
        confidence[f"{i+1}_test"] = top_softmax
        _add_stats(conversations[i],top_softmax,jsonlines_file)
    accuracy = _calculate_accuracy(labels, model_answers)
    statistics = {
                    "accuracy": accuracy,
                    "confidence_dict": confidence,
                    "model_answeres_vs_labels": list(zip(model_answers, labels))
                    }

    return statistics

def few_shot_predict():
    """
    this is a placeholder for predict function
    """
    #_prepare_testing_set()

def _calculate_accuracy(labels, model_answers):
    correct = 0.
    for i, answer in enumerate(labels):
        if  model_answers[i] == answer:
            correct += 1
    return correct/len(labels)

def _top_softmax(prob_dict, tokenizer, num_tokens=15):
    result = []
    num_tokens = min(len(prob_dict), num_tokens)
    while num_tokens:
        index = np.where(prob_dict==max(prob_dict))
        token_softmax = prob_dict[index]
        token = tokenizer.decode(index)
        result.append((token, token_softmax))
        prob_dict[index] = float("-inf")
        num_tokens -= 1
    return result
