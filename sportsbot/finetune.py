import random
#from collections import defaultdict
import os
import torch
import torch.nn.functional as F
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
from IPython import display
#from IPython.display import display as dsp
from .inference import predict

def train(
    dataset,
    validation_set=None,
    validation_labels=None,
    labels_dict=None,
    model=GPT2LMHeadModel,
    tokenizer=GPT2Tokenizer,
    batch_size=1,
    epochs=4,
    lr=2e-5,
    max_seq_len=768,
    warmup_steps=5000,
    gpt2_type="gpt2",
    device="cuda",
    output_dir=".",
    output_prefix="gpt2_fintune",
    save_model_on_epoch=True,
    eval_between_epochs=True,
    validation_file="validation",
):

    # We can add these special tokens to the vocabulary and the embeddings of the model:
    tokenizer = tokenizer.from_pretrained(gpt2_type,pad_token="<pad>", padding_side="left")
    model = model.from_pretrained(gpt2_type,pad_token_id=tokenizer.pad_token_id)
    #acc_steps = 100

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    #accumulating_batch_count = 0
    #input_tensor = None
    plt.ion()
    fig, (ax_accuracy, ax_loss, ax_labels) = plt.subplots(3, 1, figsize=(15,15), tight_layout=True)
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
        #batched_dataset = create_batches(dataset,batch_size,max_seq_len)
        batched_dataset = [conv for conv in dataset if len(conv.template) < max_seq_len]
        for index, batch in tqdm(enumerate(batched_dataset),ascii=True, desc="current batch"):

            #input_tensor = input_tensor.to(device)
            batched_tensors, labels = tokenize_data(batch, tokenizer, device)
            outputs = model(batched_tensors, labels=labels)
            loss = outputs[0]
            loss.backward()
            #loss_val = loss.item()
            #running_loss += loss_val
            logits_last_token = outputs[1][...,-2,:][:]
            current_loss_last_token = F.cross_entropy(logits_last_token.view(-1, logits_last_token.size(-1)), labels[-1].view(-1))
            running_loss += current_loss_last_token.item()

            if (index+1)%batch_size==0 or index == len(batched_dataset)-1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if index == len(batched_dataset)-1:
                acc_loss.append(running_loss/(index+1))
                running_loss = 0
                #print(f"loss on {index} iteration of the {epoch} epoch: {loss_val}")

        if eval_between_epochs:
            model.eval()
            model_stats = predict(validation_set,
                                    tokenizer,
                                    model,json_file_out=validation_file+f"_{epoch}",
                                    labels=validation_labels,
                                    labels_dict=labels_dict
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
            plot_loss_accuracy(accuracy,
                                soft_accuracy,
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
            hist_data = model_stats["hist_data"]
            plot_label_histograms(hist_data, epoch+1, lr, output_prefix)
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    #for stat in all_model_responses:
        #top_predicts = pd.DataFrame.from_dict(all_model_responses[stat])
        #print(f"epoch {stat+1}: ")
        #dsp(top_predicts)
    plt.savefig(f"loss_accuracy_graph_{output_prefix}.png")
    return model

def create_batches(data, batch_size, max_seq_len):
    data_ = sorted(data, key=lambda x: len(x.template))
    data_ = [record for record in data_ if len(record.template) <= max_seq_len]
    data_ = [data_[index:index+batch_size] for index in range(0,len(data_), batch_size)]
    random.shuffle(data_)
    return data_

def tokenize_data(conversation, tokenizer, device):
    """returns a list of tokenized torch tensors"""
    # padding_value can be whatever...
    tokens = tokenizer.encode(conversation.template)
    label_check = " " + conversation.template.split(" ")[-1]
    assert len(tokenizer.encode(label_check)) == 1
    assert label_check == tokenizer.decode(tokens[-1])
    inputs = torch.LongTensor(tokens).to(device)
    #for the labels, pad conversation part of template with -100
    conv_thread = ""
    for tweet in conversation.thread:
        conv_thread+= f"{tweet.user_handle}: {tweet.content}\n"
    label_mask = len(tokenizer.encode(conv_thread))
    token_labels = [-100. if index < label_mask else tokens[index] for index in range(len(tokens))]
    labels = torch.LongTensor(token_labels).to(device)
    return inputs, labels

def plot_loss_accuracy(updated_accuracy,
                        updated_soft_accuracy,
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
    x_axis = [count for count in range(1,epochs)]
    y_axis = updated_accuracy
    y_axis_soft = updated_soft_accuracy
    ax_accuracy.clear()
    ax_loss.clear()
    ax_labels.clear()
    ax_accuracy.set_ylim([0,1])
    ax_accuracy.set_xlabel("epoch")
    ax_accuracy.set_ylabel("accuracy")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_labels.set_xlabel("epoch")
    ax_labels.set_ylabel("avg. softmax")
    ax_labels.set_title("avg. softmax for each label")
    ax_accuracy.set_title("validation acccuracy")
    ax_loss.set_title("avg loss of final token per epoch")
    ax_accuracy.plot(x_axis, y_axis, '--ro', label='hard accuracy')
    ax_accuracy.plot(x_axis, y_axis_soft, '--yo', label='soft accuracy')
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(labels_softmaxes.keys()))))
    for _, key in enumerate(labels_softmaxes.keys()):
        next_color=next(color)
        ax_labels.plot(x_axis, labels_softmaxes[key], c=next_color, label=f"{key}") #curr_ax =
    ax_loss.plot(x_axis, updated_loss,'--ro', label='training')
    ax_loss.plot(x_axis, validation_loss, '--bo', label='validation')
    ax_loss.legend(loc="best")
    ax_accuracy.legend(loc="best")
    ax_labels.legend(loc="best")
    display.clear_output(wait=True)
    display.display(plt.gcf())

def plot_label_histograms(hist_data, epoch, lr, output_prefix):
    fig_2, ax_hist = plt.subplots(tight_layout=True)
    width = 0.35
    ground_truth, model_predictions = hist_data[0], hist_data[1]
    ground_truth["invalid"] = 0
    labels = list(ground_truth.keys())
    for key in labels:
        if key not in model_predictions.keys():
            model_predictions[key] = 0
    for key in model_predictions.keys():
        if key not in labels:
            labels.append(key)
            ground_truth[key] = 0
    x_ax = np.arange(len(labels)) #label locations
    count = max(len(ground_truth.values()), len(model_predictions.values()))
    assert len(ground_truth.keys()) == len(model_predictions.keys()), f"{ground_truth.keys()} {model_predictions.keys()}"

    model_predictions = [model_predictions[val] for val in sorted(model_predictions.keys())]
    ground_truth = [ground_truth[val] for val in sorted(ground_truth.keys())]

    ground_truth_plt = ax_hist.bar(x_ax-width/2, ground_truth, width, label='ground truth')
    model_predictions_plt = ax_hist.bar(x_ax+width/2, model_predictions, width, label='model predictions')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax_hist.set_ylabel('num. words')
    num_labels = [i for i in range(count+1)]
    ax_hist.set_yticks(num_labels)
    ax_hist.set_yticklabels(num_labels)
    ax_hist.set_title(f'ground truth vs. predicted words for epoch {epoch-1}')
    ax_hist.set_xticks(x_ax)
    ax_hist.set_xticklabels(sorted(labels))
    ax_hist.legend()
    autolabel(ground_truth_plt, ax_hist)
    autolabel(model_predictions_plt, ax_hist)
    plt.savefig(f"label_hist_{output_prefix}_{lr}_{epoch}.png")
    plt.close(fig=fig_2)


def autolabel(rects, axis):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axis.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def pil_grid(images, max_horiz=np.iinfo(int).max, file_name='image.png'):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes = [0] * n_horiz
    v_sizes = [0] * (n_images // n_horiz) if not n_images // n_horiz else  [0] * ((n_images // n_horiz)+1)
    for index, image in enumerate(images):
        height, width = index % n_horiz, index // n_horiz
        h_sizes[height] = max(h_sizes[height], image.size[0])
        v_sizes[width] = max(v_sizes[width], image.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for index, image in enumerate(images):
        im_grid.paste(image, (h_sizes[index % n_horiz], v_sizes[index // n_horiz]))
    im_grid.save(file_name)
    return im_grid
