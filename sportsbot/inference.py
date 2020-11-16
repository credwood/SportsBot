"""
Functions for testing Hugginface's GPT-2 trained model
"""
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .datasets import _prepare_few_shot_testing_set, _save_data, ConversationPrompt

def download_model_tokenizer(model_type="gpt2"):
    """
    colab can run GPT2 mdoels: 'gpt2', 'gpt2-medium, 'gpt2-large'
    """
    tokenizer_instantiate = GPT2Tokenizer.from_pretrained(model_type)
    tokenizer_instantiate.pad_token = tokenizer_instantiate.eos_token

    # add the EOS token as PAD token to avoid warnings
    model_instantiate = GPT2LMHeadModel.from_pretrained(model_type,
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
        input_ids = tokenizer.encode(tweets,return_tensors='pt')
        output = model(input_ids)
        predicted_prob = F.softmax(output.logits[0, -1, :], dim=-1) #add for batch size
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
            device="cuda",
            num_top_softmax=20,
            json_file_out='add_stats_output.jsonl',
            labels=None
        ):
    """
    Function for finetuned model. saves and returns `Conversation` objects
    with model statistics (top SoftMax values for each conversation).
    """
    model.eval()
    model.to(device)
    conversations = defaultdict()
    answers = []
    for i, test_conv in enumerate(test_convs):
        tweet_template = test_conv
        input_ids = tokenizer.encode(tweet_template)
        input_tensor = torch.LongTensor(input_ids).to(device)
        output = model(input_tensor,return_dict=True)
        logits = output.logits
        logits = logits[...,-1,:]
        predicted_prob = F.softmax(logits, dim=-1)
        top_softmax = _top_softmax(predicted_prob, tokenizer, num_top_softmax)
        answers.append(top_softmax[0][0])
        if labels:
            softmax_tokens = _label_softmax(predicted_prob, tokenizer, [' No',' N/A',' Unlikely',' Maybe',' Probably',' Yes'])
            conversations[i] = [test_conv, softmax_tokens, labels[i], top_softmax]
        else:
            softmax_tokens = _label_softmax(predicted_prob, tokenizer, [' No',' N/A',' Unlikely',' Maybe',' Probably',' Yes'])
            conversations[i] = [test_conv, softmax_tokens, top_softmax]
    if labels:
        conversations["accuracy"] = str(_calculate_accuracy(labels, answers))
    with open(json_file_out, "w") as dump:
        json.dump(conversations, dump, indent=4)
    return conversations

def _calculate_accuracy(labels, model_answers):
    correct = 0.
    for i, answer in enumerate(labels):
        if  model_answers[i] == answer:
            correct += 1
    return correct/len(labels)

def _top_softmax(prob_dict, tokenizer, num_tokens):
    num_tokens = min(len(prob_dict), num_tokens)
    _, sorted_indices = torch.sort(prob_dict[:], descending=True)
    sorted_indices = list(sorted_indices.detach().numpy())
    return [(tokenizer.decode([index]), str(prob_dict[index])) for index in sorted_indices[:num_tokens]]

def _label_softmax(prob_dict, tokenizer, labels_lst):
    prob_dict = list(prob_dict.detach().numpy())
    return [{label: str(prob_dict[tokenizer.encode(label)[0]])} for label in labels_lst]
