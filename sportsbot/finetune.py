"""
Functions for fine-tuning GPT2 models on Twitter conversations or any foreign data.
Includes visualization functions as well.
"""
import random
from collections import defaultdict
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
from IPython import display
#from IPython.display import display as dsp
from .inference import predict
from .label_dictionaries import classes_dict, label_dict

def train(
    dataset,
    question,
    validation_set=None,
    validation_labels=None,
    labels_dict=label_dict, #defaults to global label_dict but better to use customized
    model=GPT2LMHeadModel,
    tokenizer=GPT2Tokenizer,
    batch_size=5,
    epochs=4,
    lr=2e-5,
    max_seq_len=1024,
    warmup_steps=5000,
    gpt2_type="gpt2",
    device="cuda",
    output_dir=".",
    output_prefix="gpt2_fintune",
    save_model_on_epoch=True,
    eval_between_epochs=True,
    validation_file="validation",
    download=True,
    foreign_data=False,
    plot_loss=True,
    prompt=None,
):
    """
    Please see README for more detailed parameter comments
    Training loop for `Conversation` objects or foreign data. Hugging face's GPT2 transformer
    models
    """
    tokenizer = tokenizer.from_pretrained(gpt2_type,pad_token="<pad>", padding_side="left") if download else tokenizer
    model = model.from_pretrained(gpt2_type,pad_token_id=tokenizer.pad_token_id) if download else model
    #acc_steps = 100

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    dataset = [conv for conv in dataset if len(tokenizer.encode(conv.template)) < max_seq_len] if not foreign_data else dataset
    #accumulating_batch_count = 0
    plt.ion()
    fig, (ax_accuracy, ax_loss, ax_labels) = plt.subplots(3, 1, figsize=(20,25), tight_layout=True)
    accuracy = []
    soft_accuracy = []
    validation_loss = []
    acc_loss = []
    running_loss = 0
    #label_softmax = []
    #all_model_responses = defaultdict()
    running_label_loss = {
                  " No": [],
                  " Remote": [],
                  " Unsure": [],
                  " Probably": [],
                  " Yes": [],
                  " Neutral": [],
                  " None": [],
                  " Positive": [],
                  " Defensive": [],
                  " Negative": [],
                  " Opposition": [],
                  " Discussion": [],
                  " Agreement": []
    }
    for epoch in range(epochs):

        model.train()

        random.shuffle(dataset)

        print(f"Training epoch {epoch+1}")
        
        for index, batch in tqdm(enumerate(dataset),ascii=True, desc="current batch"):

            batched_tensors, labels = tokenize_data(batch, tokenizer, device, foreign_data, prompt)
            outputs = model(batched_tensors, labels=labels)
            loss = outputs[0]
            loss.backward()
            loss_val = loss.item()
            running_loss += loss_val
            logits_last_token = outputs[1][...,-2,:][:]
            current_loss_last_token = F.cross_entropy(logits_last_token.view(-1, logits_last_token.size(-1)), labels[-1].view(-1))
            running_loss += current_loss_last_token.item()

            if (index+1)%batch_size==0 or index == len(dataset)-1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if index == len(dataset)-1:
                acc_loss.append(running_loss/(index+1))
                running_loss = 0
                #print(f"loss on {index} iteration of the {epoch} epoch: {loss_val}")

        if eval_between_epochs:
            model.eval()
            model_stats = predict(validation_set,
                                    tokenizer,
                                    model,
                                    json_file_out=validation_file+f"_{epoch}",
                                    labels=validation_labels,
                                    labels_dict=labels_dict,
                                    foreign_data=foreign_data
                          )
            #model_responses = defaultdict()
            #model_responses["correct answer"] = [labels_dict["all_values"][model_stats[str(index)][2]] for index in range(len(validation_set))]
            #model_responses["1st choice"] = [model_stats[str(index)][3][0][0] for index in range(len(validation_set))]
            #model_responses["2nd choice"] = [model_stats[str(index)][3][1][0] for index in range(len(validation_set))]
            #model_responses["3rd choice"] = [model_stats[str(index)][3][2][0] for index in range(len(validation_set))]
            #all_model_responses[epoch] = model_responses
            accuracy.append(float(model_stats["accuracy"]))
            soft_accuracy.append(float(model_stats["soft_accuracy"]))
            validation_loss.append(float(model_stats["validation_loss"]))
            label_softmaxes = model_stats["label_softmaxes"]
            for label in label_softmaxes.keys():
                running_label_loss[label].append(label_softmaxes[label])
            #top_model_predict = [int(model_stats[str(index)][3][0][1]) for index in range(len(validation_set))]
            #labels_top_model_predict = [model_stats[str(index)][3][0][0] for index in range(len(validation_set))]
            #correct_answers = [labels_dict["all_values"][model_stats[str(index)][2]] for index in range(len(validation_set))]
            #hist_data = model_stats["hist_data"]
            ground_truth = [labels_dict["all_values"][model_stats[str(index)][2]] for index in range(len(validation_set))]
            model_predictions = [model_stats[str(index)][3][0][0] for index in range(len(validation_set))]
            if plot_loss:
                plot_loss_accuracy(accuracy,
                                    soft_accuracy,
                                    labels_dict["baseline_accuracy"],
                                    acc_loss,
                                    validation_loss,
                                    ax_accuracy,
                                    ax_loss,
                                    ax_labels,
                                    epoch+2,
                                    fig,
                                    running_label_loss,
                                    labels_dict=labels_dict
                )
            create_confusion_matrix(ground_truth, model_predictions, question, epoch+1, lr, output_prefix)

        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    #for stat in all_model_responses:
        #top_predicts = pd.DataFrame.from_dict(all_model_responses[stat])
        #print(f"epoch {stat+1}: ")
        #dsp(top_predicts)
    if plot_loss:
        fig.savefig(f"loss_accuracy_graph_{output_prefix}.png")
    return model

def create_batches(data, batch_size, max_seq_len):
    """batches dataset"""
    data_ = sorted(data, key=lambda x: len(x.template))
    data_ = [record for record in data_ if len(record.template) <= max_seq_len]
    data_ = [data_[index:index+batch_size] for index in range(0,len(data_), batch_size)]
    random.shuffle(data_)
    return data_

def tokenize_data(conversation, tokenizer, device, foreign_data, prompt):
    """returns two lists of tokenized torch tensors: inputs and labels"""
    tokens = tokenizer.encode(conversation.template) if not foreign_data else tokenizer.encode(conversation)
    label_check = " " + conversation.template.split(" ")[-1] if not foreign_data else " " + conversation.split(" ")[-1]
    assert len(tokenizer.encode(label_check)) == 1
    assert label_check == tokenizer.decode(tokens[-1])
    inputs = torch.LongTensor(tokens).to(device)
    #for the labels, pad conversation part of template with -100
    conv_thread = ""
    if not foreign_data:
        for tweet in conversation.thread:
            conv_thread+= f"{tweet.user_handle}: {tweet.content}\n"
        label_mask = len(tokenizer.encode(conv_thread))
        token_labels = [-100. if index < label_mask else tokens[index] for index in range(len(tokens))]
        labels = torch.LongTensor(token_labels).to(device)
    else:
        len_input = len(tokens) - len(tokenizer.encode(prompt)) -1
        token_labels = [-100. if index < len_input else tokens[index] for index in range(len(tokens))]
        labels = torch.LongTensor(token_labels).to(device)
    return inputs, labels

def plot_loss_accuracy(updated_accuracy,
                        updated_soft_accuracy,
                        base_accuracy,
                        updated_loss,
                        validation_loss,
                        ax_accuracy,
                        ax_loss,
                        ax_labels,
                        epochs,
                        fig,
                        labels_softmaxes,
                        labels_dict=None
):
    """ 
    plots loss and accuracy graphs, updated each epoch.
    saves plot when finetuning is done
    """
    x_axis = [count for count in range(1,epochs)]
    ax_accuracy.clear()
    ax_loss.clear()
    ax_labels.clear()
    ax_accuracy.set_ylim([0,1])
    ax_labels.set_xlabel("epoch", fontsize=15)
    ax_labels.set_ylabel("avg. softmax", fontsize=15)
    ax_loss.set_xlabel("epoch", fontsize=15)
    ax_loss.set_ylabel("loss", fontsize=15)
    ax_accuracy.set_xlabel("epoch", fontsize=15)
    ax_accuracy.set_ylabel("accuracy", fontsize=15)
    ax_labels.set_title("avg. softmax for each label")
    ax_accuracy.set_title("validation acccuracy")
    ax_loss.set_title("avg loss of final token per epoch")
    ax_accuracy.plot(x_axis, updated_accuracy, '--ro', label='hard accuracy')
    ax_accuracy.plot(x_axis, updated_soft_accuracy, '--yo', label='soft accuracy')
    base_plot = [base_accuracy for _ in range(1, epochs)]
    ax_accuracy.plot(x_axis, base_plot, '--go', label='arg. max accuracy')
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(labels_softmaxes.keys()))))
    for index, key in enumerate(labels_softmaxes.keys()):
        next_color=next(color)
        curr_ax = ax_labels.plot(x_axis, labels_softmaxes[key], c=next_color, label=f"{key}")
    ax_loss.plot(x_axis, updated_loss,'--ro', label='training')
    ax_loss.plot(x_axis, validation_loss, '--bo', label='validation')
    ax_loss.legend()
    ax_accuracy.legend()
    ax_labels.legend()
    #plt.close()
    #display.clear_output(wait=True)
    #display.display(plt.gcf())

def create_confusion_matrix(
                            y_true,
                            y_pred,
                            question,
                            epoch=None,
                            lr=None,
                            output_prefix=None,
                            classes = classes_dict,
                            out_file="confusion.png"
):
    """ 
    creates and plots a confusion matrix given two list (ground_truths and model predictions)
    :param list y_true: list of all ground truths
    :param list y_pred: list of all model predictions
    :param dict classes: dict of classes, converts naturl language response to indexing integer
    """
    y_pred = [prediction if prediction in classes[question].keys() else "invalid" for prediction in y_pred]
    amount_classes = len(list(classes[question].keys()))

    confusion_matrix = pd.DataFrame(np.zeros(shape=[amount_classes, amount_classes]),
                                    columns=list(classes[question].keys()))
    for index in range(len(y_true)):
        target = y_true[index]

        output = y_pred[index]

        confusion_matrix[target][classes[question][output]] += 1
    sn.set(rc={'figure.figsize':(11.7,8.27)})
    sn.set(font_scale=2.0) # for label size
    plt.figure()
    confusion_matrix = confusion_matrix/confusion_matrix.sum()
    confusion_matrix.round(3)
    sn.heatmap(confusion_matrix,
                xticklabels=classes[question].keys(),
                yticklabels=classes[question].keys(),
                annot=True,
                annot_kws={"size": 16},
                fmt=".3"
    ) # font size
    #plot.figure.show()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if epoch is not None:
        plt.savefig(f"confusion_matrix_{output_prefix}_{lr}_{epoch}.png")
        #display.display(plt.gcf())
        #plt.close()
    else:
        plt.savefig(out_file)

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def pil_grid(images, max_horiz=np.iinfo(int).max, file_name='image.png'):
    """returns images as a single grided image"""
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes = [0] * n_horiz
    v_sizes = [0] * (n_images // n_horiz) if not n_images // n_horiz else  [0] * ((n_images // n_horiz)+1)
    for index, img in enumerate(images):
        height, vert = index % n_horiz, index // n_horiz
        h_sizes[height] = max(h_sizes[height], img.size[0])
        v_sizes[vert] = max(v_sizes[vert], img.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for index, img in enumerate(images):
        im_grid.paste(img, (h_sizes[index % n_horiz], v_sizes[index // n_horiz]))
    im_grid.save(file_name)
    return im_grid
