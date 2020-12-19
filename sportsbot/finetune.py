import random
from collections import defaultdict
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from IPython import display
from IPython.display import display as dsp
#import torch.nn.functional as F
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
    validation_file="validation"
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
    fig, (ax_accuracy, ax_loss) = plt.subplots(1, 2, tight_layout=True)
    accuracy = []
    validation_loss = []
    acc_loss = []
    running_loss = 0
    all_model_responses = defaultdict()
    for epoch in range(epochs):

        model.train()
        random.shuffle(dataset)

        print(f"Training epoch {epoch}")
        #batched_dataset = create_batches(dataset,batch_size,max_seq_len)
        batched_dataset = [conv for conv in dataset if len(conv.template) < max_seq_len]
        for index, batch in tqdm(enumerate(batched_dataset),ascii=True, desc="current batch"):

            #input_tensor = input_tensor.to(device)
            batched_tensors, labels = tokenize_data(batch, tokenizer, device)
            outputs = model(batched_tensors, labels=labels)
            loss = outputs[0]
            loss.backward()
            loss_val = loss.item()
            running_loss += loss_val

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
            model_responses = defaultdict()
            model_responses["correct answer"] = [labels_dict[model_stats[str(index)][2]] for index in range(len(validation_set))]
            model_responses["1st choice"] = [model_stats[str(index)][3][0][0] for index in range(len(validation_set))]
            model_responses["2nd choice"] = [model_stats[str(index)][3][1][0] for index in range(len(validation_set))]
            model_responses["3rd choice"] = [model_stats[str(index)][3][2][0] for index in range(len(validation_set))]
            all_model_responses[epoch] = model_responses
            accuracy.append(float(model_stats["accuracy"]))
            validation_loss.append(float(model_stats["validation_loss"]))
            plot_loss_accuracy(accuracy, acc_loss, validation_loss, ax_accuracy, ax_loss, epoch+2, fig)
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    for stat in all_model_responses:
        top_predicts = pd.DataFrame.from_dict(all_model_responses[stat])
        print(f"epoch {stat+1}: ")
        dsp(top_predicts)
    plt.savefig(f"loss_accuracy_graph_{output_prefix}.png")
    return model

def create_batches(data, batch_size,max_seq_len):
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

def plot_loss_accuracy(updated_accuracy, updated_loss, validation_loss, ax_accuracy, ax_loss, epochs, fig):
    x_axis = [count for count in range(1,epochs)]
    y_axis = updated_accuracy
    ax_accuracy.clear()
    ax_loss.clear()
    ax_accuracy.set_xlabel("epoch")
    ax_accuracy.set_ylabel("accuracy")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_accuracy.set_title("validation acccuracy")
    ax_loss.set_title("loss")
    ax_accuracy.plot(x_axis, y_axis, '--ro')
    ax_loss.plot(x_axis, updated_loss,'--ro', label='training')
    ax_loss.plot(x_axis, validation_loss, '--bo', label='validation')
    ax_loss.legend(loc="best")
    display.clear_output(wait=True)
    display.display(plt.gcf())
