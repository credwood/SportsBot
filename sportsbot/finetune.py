import random
import os
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
#import torch.nn.functional as F
from .inference import predict

def train(
    dataset,
    validation_set=None,
    validation_labels=None,
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
    test_mode=False,
    save_model_on_epoch=True,
    eval_between_epochs=True
):

    # We can add these special tokens to the vocabulary and the embeddings of the model:
    tokenizer = tokenizer.from_pretrained(gpt2_type,pad_token="<pad>")
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

    for epoch in range(epochs):

        random.shuffle(dataset)

        print(f"Training epoch {epoch}")
        batched_dataset = create_batches(dataset,batch_size,max_seq_len)
        for index, batch in tqdm(enumerate(batched_dataset),ascii=True, desc="current batch"):
            """
            stuffing tensors:
                (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 1024)

                if carry_on and idx != len(train_dataloader) - 1:
                    continue
            """
            #input_tensor = input_tensor.to(device)
            batched_tensors, labels, mask = tokenize_data(batch, tokenizer, device)
            outputs = model(batched_tensors, attention_mask=mask, labels=labels)
            loss = outputs[0]
            loss.backward()

            if (index+1)%16 or index == len(dataset)-1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

        if eval_between_epochs:
            predict(validation_set,tokenizer, model,json_file_out=f"validation_{epoch}.json",labels=validation_labels)
            #accumulating_batch_count += 1
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

def create_batches(data, batch_size,max_seq_len):
    data = sorted(data, key=lambda x: len(x.template))
    data = [record for record in data if len(record.template) <= max_seq_len]
    data = [data[index:index+batch_size] for index in range(0,len(data), batch_size)]
    random.shuffle(data)
    return data

def tokenize_data(conversations, tokenizer, device):
    """returns a list of tokenized torch tensors"""
    # padding_value can be whatever...
    tokens = [[tokenizer.encode(conv.template) for conv in conversations] ]
    inputs = pad_sequence([torch.LongTensor(template) for template in tokens], padding_value=tokenizer.pad_token_id).to(device)
    # 1 for real tokens and 0 for padded tokens
    mask = (inputs != tokenizer.pad_token_id).float()
    #for the labels, pad conversation part of template with -100
    label_mask = []
    for conversation in conversations:
        conv_thread = ""
        for tweet in conversation.thread:
            conv_tread+= f"{tweet.user_handle}: {tweet.content}\n"
        label_mask.append(len(tokenizer.encode(conv_thread)))
    token_labels = [[-100 if index < label_mask[i] else tokens[i][index] for index in range(len(tokens[i]))] for i in range(len(tokens))]
    labels = pad_sequence([torch.LongTensor(template) for template in token_labels], padding_value=tokenizer.pad_token_id).to(device)
    return inputs, labels, mask


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[0] + packed_tensor.size()[0] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor])
        return packed_tensor, True, None
