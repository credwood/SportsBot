"""
Functions for testing Hugginface's GPT-2 trained model
"""
import json
from collections import defaultdict, Counter
from numpy import exp, log
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def predict(test_convs,
            tokenizer,
            model,
            device="cuda",
            num_top_softmax=20,
            json_file_out='add_stats_output.jsonl',
            labels=None,
            labels_dict=None,
        ):
    """
    Function for finetuned model. saves and returns `Conversation` objects
    with model statistics (top SoftMax values for each conversation).
    """
    assert  sum([len(tokenizer.encode(label)) == 1 for label in labels_dict["all_values"].values()]) == len(list(labels_dict["all_values"].values()))
    model.to(device)
    model.eval()
    conversations = defaultdict()
    answers = []
    accumulated_loss = []
    dataset_size = len(test_convs)
    running_label_loss = {
                  " No": 0,
                  " Remote": 0,
                  " Unsure": 0,
                  " Probably": 0,
                  " Yes": 0,
                  " Neutral": 0,
                  " None": 0,
                  " Positive": 0,
                  " Defensive": 0,
                  " Negative": 0,
                  " Opposition": 0,
                  " Discussion": 0,
                  " Agreement": 0
    }
    for i, test_conv in enumerate(test_convs):
        tweet_template = test_conv.template
        input_ids = tokenizer.encode(tweet_template)
        label = labels_dict["all_values"][labels[i]]
        label = tokenizer.encode(label)
        input_tensor = torch.LongTensor(input_ids).to(device)
        label = torch.LongTensor(label).to(device)
        #test_conv.template += labels_dict[labels[i]]
        #tensor, labels, _ = tokenize_data(test_conv, tokenizer, device)
        with torch.no_grad():
            output = model(input_tensor,labels=input_tensor,return_dict=True)
        logits = output.logits
        logits = logits[...,-1,:]
        accumulated_loss.append(F.cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1)).item())
        predicted_prob = F.softmax(logits, dim=-1)
        top_softmax = _top_softmax(predicted_prob, tokenizer, num_top_softmax)
        answers.append(top_softmax[0][0])
        if labels is not None:
            all_label_softmax = _all_label_softmax(predicted_prob, tokenizer, labels_dict["all_values"])
            conversations[str(i)] = [test_conv.template, all_label_softmax, labels[i], top_softmax]
        else:
            conversations[str(i)] = [test_conv.template, top_softmax]
    if labels is not None:
        conversations["accuracy"] = str(_calculate_accuracy(labels, answers, labels_dict["all_values"]))
        conversations["soft_accuracy"] = str(_soft_accuracy(labels, labels_dict["bucketed_labels"], answers))
        conversations["validation_loss"] = sum(accumulated_loss)/dataset_size
        answers_str = [answer if answer in labels_dict["all_values"].values() else "invalid" for answer in answers]
        labels_str = [labels_dict["all_values"][label] for label in labels]
        conversations["hist_data"] = [Counter(labels_str), Counter(answers_str)]
        running_label_loss = _label_softmax(predicted_prob, tokenizer, labels_dict["all_values"], running_label_loss)
        for label in labels_dict["all_values"].values():
            running_label_loss[label] = running_label_loss[label]/dataset_size
        conversations["label_softmaxes"] = running_label_loss
        print(f"accuracy: {conversations['accuracy']}")
    with open(json_file_out, "w") as dump:
        json.dump(conversations, dump, indent=4)
    return conversations

def _calculate_accuracy(labels, model_answers, labels_dict):
    correct = 0.
    for i, answer in enumerate(labels):
        if  labels_dict[answer] == model_answers[i]:
            correct += 1
    return correct/len(labels)

def _top_softmax(prob_dict, tokenizer, num_tokens):
    num_tokens = min(len(prob_dict), num_tokens)
    _, sorted_indices = torch.sort(prob_dict[:], descending=True)
    sorted_indices = list(sorted_indices.cpu().numpy())
    return [(tokenizer.decode([index]), str(prob_dict[index].item())) for index in sorted_indices[:num_tokens]]

def _label_softmax(prob_dict, tokenizer, labels_dict, running_label_loss):
    prob_dict = list(prob_dict.detach().cpu().numpy())
    return_dict = defaultdict()
    for label in labels_dict.values():
        running_label_loss[label] += prob_dict[tokenizer.encode(label)[0]]
    return running_label_loss

def _all_label_softmax(prob_dict, tokenizer, labels_dict):
    prob_dict = list(prob_dict.detach().cpu().numpy())
    return [{label: str(prob_dict[tokenizer.encode(label)[0]].item())} for label in labels_dict.values()]

#log prob(token_j)  = logit_j - log(sum_k exp(logit_k))
def _log_prob(logits_tensor, labels_lst, num_tokens, tokenizer):
    _, sorted_indices = torch.sort(logits_tensor[:], descending=True)
    sorted_indices = list(sorted_indices.detach().numpy())
    logits = logits_tensor.detach.numpy()
    log_probs = list(logits - log(sum(exp(logits))))
    #converting logits into log probs shouldn't change indices...
    return [(tokenizer.decode([index]), str(log_probs[index].item())) for index in sorted_indices[:num_tokens]]

def _soft_accuracy(labels, bucketed_labels, model_answers):
    correct = 0.
    for i, answer in enumerate(labels):
        if model_answers[i] in bucketed_labels[answer]:
            correct += 1
    return correct/len(labels)
