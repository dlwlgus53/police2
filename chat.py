# load moduels
# load model for classification
# load model for generation


# load model
# load tokenizer

# read example.json
# loop over the turns

# for each turn, classify if it has information or not
# if has information, generate response
# if not, continue


# save the response to the json file

# https://towardsdatascience.com/how-to-train-bert-for-q-a-in-any-language-63b62c780014
import torch
from dataset import QA_dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast
import argparse
import torch.nn as nn
import init
import logging

import pdb
parser = argparse.ArgumentParser()

'''training'''

'''enviroment'''
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')

parser.add_argument('--test_path',  type=str,
                    default='../example.json')

args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")


if __name__ == "__main__":
    logger.info(args)
    train_dataset = QA_dataset(args.train_path)
    dev_dataset = QA_dataset(args.dev_path)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             batch_size=args.test_batch_size,
                                             shuffle=True)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForQuestionAnswering.from_pretrained(args.base_trained)
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    tokenizer = BertTokenizerFast.from_pretrained(args.base_trained)
    optim = AdamW(model.parameters(), lr=5e-5)

    min_loss = 100000
    p = 0

    for epoch in range(args.max_epoch):
        for idx, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)

            loss = outputs[0].sum()
            loss.backward()
            optim.step()

            if idx % 10 == 0:
                logger.info(f"Idx {idx}, loss : {loss.item()}")
                start_idx = torch.argmax(outputs.start_logits, axis=-1)
                end_idx = torch.argmax(outputs.end_logits, axis=-1)
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[0][start_idx[0]:end_idx[0]+1]))
                label = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
                    input_ids[0][start_positions[0]:end_positions[0]+1]))

                logger.info(f"Label : {label}")
                logger.info(f"Pred : {answer}")

        for idx, batch in enumerate(dev_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)

            loss = outputs[0].sum()

        if min_loss > loss:
            min_loss = loss

            torch.save(model.state_dict(), f'./model/{args.save_prefix}.pt')
            logger.info("safely saved")
            p = 0
        else:
            p += 1
        if p == args.patient:
            logger.info("early stop")
            break
