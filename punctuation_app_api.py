import pickle

import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from dictabert import DictaAutoTokenizer


def tokenize_text(text, tokenizer):
    return tokenizer.tokenize(text)


def get_gpu_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def divide_chunks(text, n):
    # looping till length l
    for i in range(0, len(text), n):
        yield text[i:i + n]


class PunctuationAPI:
    def __init__(self, chunk_size):
        self.batch_size = 32
        model_name = 'ABG2.0NoUnkBlank6P/ckpt_48800'
        self.tokenizer = DictaAutoTokenizer.from_pretrained(model_name)

        self.chunk_size = chunk_size
        model_name = 'ABG2.0NoUnkBlank6P/ckpt_48800'
        self.tokenizer = DictaAutoTokenizer.from_pretrained(model_name)

        self.model = pickle.load(open('models/fine_tuned_model_1000.p', 'rb'))

        self.device = get_gpu_device()

        self.df = None
        self.dataset = None
        self.dataloader = None

        self.results = None
        self.generated_text = []

    def get_data_loaders(self):
        # For validation the order doesn't matter, so we'll just read them sequentially.
        dataloader = DataLoader(
            self.dataset,  # The validation samples.
            sampler=SequentialSampler(self.dataset),  # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )

        return dataloader

    def preprocess_text_data(self, text_input):
        self.df = {'chk_idx': [], 'tokens': [], 'attention_mask': []}
        chunk_indices = []
        tokens = []
        attention_masks = []
        # drop punctuations and label the data
        curr_tokens = tokenize_text(text_input, self.tokenizer)
        # drop punctuations
        curr_tokens = [t for t in curr_tokens if t not in ['.', '!', ',', ':', ';', '?']]
        if len(curr_tokens) > self.chunk_size - 2:
            chunks = list(divide_chunks(curr_tokens, self.chunk_size - 2))
            del chunks[-1]
            for chk_idx, chunk_tokens in enumerate(chunks):
                chunk_indices.append(chk_idx)
                tokens.append(self.tokenizer.encode(chunk_tokens, return_tensors='pt'))
                attention_masks.append(torch.tensor([[1] * 128]))
                # labelling [CLS], [SEP]

            self.df['chk_idx'].extend(chunk_indices)
            self.df['tokens'].extend(tokens)
            self.df['attention_mask'].extend(attention_masks)

            self.df = pd.DataFrame(self.df)

            input_ids = list(self.df['tokens'].values)
            attention_masks = list(self.df['attention_mask'].values)

            # Convert the lists into tensors.
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            # Combine the inputs into a TensorDataset.
            self.dataset = TensorDataset(input_ids, attention_masks)

            self.dataloader = self.get_data_loaders()

    def evaluate(self, text_input):
        self.preprocess_text_data(text_input)
        all_tokens = []
        all_predictions = []

        all_blank_score = []
        all_period_score = []
        all_comma_score = []
        all_question_mark_score = []

        self.model.eval()

        for step, batch in enumerate(self.dataloader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)

            with torch.no_grad():
                result = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

            logit = result.logits

            logit = logit.detach().cpu().numpy()

            # save the predictions
            tokens = self.tokenizer.convert_ids_to_tokens(b_input_ids.flatten())
            predictions = list(np.argmax(logit, axis=2).flatten())

            all_tokens.extend(tokens)
            all_predictions.extend(predictions)
            # 'BLANK', 'PERIOD', 'COMMA', 'QUESTION MARK'
            flatten_logit = logit.reshape(-1, 4)
            all_blank_score.extend(flatten_logit[:, 0])
            all_period_score.extend(flatten_logit[:, 1])
            all_comma_score.extend(flatten_logit[:, 2])
            all_question_mark_score.extend(flatten_logit[:, 3])

        self.results = pd.DataFrame({
            'token': all_tokens,
            'prediction': all_predictions,
            'bank_score': all_blank_score,
            'period_score': all_period_score,
            'comma_score': all_comma_score,
            'question_mark': all_question_mark_score,
        })

        self.generate_punctuated_text()

    def generate_punctuated_text(self):
        for i in range(len(self.results) - 1):
            row = self.results.iloc[i]
            next_row = self.results.iloc[i + 1]

            if row['token'] == '[CLS]' or row['token'] == '[SEP]':
                continue

            token = row['token']
            if row['token'][0] == '#' and row['token'][1] == '#':
                token = token[2:]
            self.generated_text.append(token)

            if next_row['token'][0] == '#' and next_row['token'][1] == '#':
                continue

            if row['prediction'] == 0:
                self.generated_text.append(' ')
            elif row['prediction'] == 1:
                self.generated_text.append('. ')
            elif row['prediction'] == 2:
                self.generated_text.append(', ')
            elif row['prediction'] == 3:
                self.generated_text.append('? ')

        self.generated_text = ''.join(self.generated_text)
