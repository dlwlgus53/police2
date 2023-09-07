from datasets import load_dataset
from tqdm.auto import tqdm  # for showing progress bar
from transformers import BertTokenizerFast
import json
import pdb
import ontology
import logging


import torch
logger = logging.getLogger("my")


class QA_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        logger.info("read data")
        raw_data = json.load(open(data_path, "r"))
        self.tokenizer = BertTokenizerFast.from_pretrained(
            'kykim/bert-kor-base')
        context, question, answer = self.seperate_data(raw_data)

        answer = self.add_answer_info(context, answer)
        emb_con_ques = self.tokenizer(context, question,
                                      truncation=True, padding='max_length',
                                      max_length=128, return_tensors='pt')

        self.add_token_positions(emb_con_ques, answer)
        self.dataset = emb_con_ques

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.dataset.items()}

    def __len__(self):
        return len(self.dataset.input_ids)

    def seperate_data(self, dataset):
        question = []
        answer = []
        context = []

        for i, dialogue in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"seperate data {i}/{len(dataset)}")
            d_id = dialogue['ID']
            dialogue_text = ""
            for t_id, turn in enumerate(dialogue['log']):
                # TODO
                for key_idx, key in enumerate(ontology.QA['all-domain']):
                    q = ontology.QA[key]['description']
                    # only casese when the answer is exist # 질문도 합쳐서
                    if key in turn['belief']:
                        a = turn['belief'][key]

                        answer.append(str(a))
                        question.append(q)
                        context.append(turn['criminal'])

        return context, question, answer

    def add_token_positions(self, encodings, answers):
        # initialize lists to contain the token indices of answer start/end
        start_positions = []
        end_positions = []
        logger.info("add token position, It takes a few minutes")
        for i in range(len(answers)):
            if i % 10 == 0:
                logger.info(f"add_token_positions {i}/{len(answers)}")
            # append start/end token position using char_to_token method
            start_positions.append(encodings.char_to_token(
                i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(
                i, answers[i]['answer_end']))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            # end position cannot be found, char_to_token found space, so shift position until found
            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(
                    i, answers[i]['answer_end'] - shift)
                shift += 1
        # update our encodings object with the new token-based start/end positions
        encodings.update({'start_positions': start_positions,
                         'end_positions': end_positions})

    def add_answer_info(self, context, answer):
        new_answer = []
        logger.info("add answer info, It takes a few minutes")
        for i, (c, a) in enumerate(zip(context, answer)):
            if i % 10 == 0:
                logger.info(f"add_answer_info {i}/{len(answer)}")
            answer_start = c.find(a)
            temp = {'text': c, 'answer_start': answer_start,
                    'answer_end': answer_start + len(a)}
            new_answer.append(temp)
        return new_answer


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('kykim/bert-kor-base')
    train_dataset = QA_dataset(
        "/home/jihyunlee/police/POLICE_data/all_data.json")
    loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=16,
                                         shuffle=True)
    loop = tqdm(loader)
    device = 'cuda'
    for batch in loop:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        pdb.set_trace()
