import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification
import math

fchidden = 256
hiddendim_lstm = 128
embeddim = 768
numlayers = 12


class Bert_Base(nn.Module):
    def __init__(self, numclasses):
        super(Bert_Base, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.dropout = nn.Dropout(0.1)

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', # noqa
                                                                   output_hidden_states=False, # noqa
                                                                   output_attentions=False, # noqa
                                                                   num_labels=self.numclasses) # noqa
        print("BERT-single Model Loaded")

    def forward(self, inp_ids, att_mask, token_ids, labels):
        loss, out = self.bert(input_ids=inp_ids, attention_mask=att_mask,
                              token_type_ids=token_ids, labels=labels)
        return loss, out


class Bert_LSTM(nn.Module):
    def __init__(self, numclasses):
        super(Bert_LSTM, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.hiddendim_lstm = hiddendim_lstm
        self.dropout = nn.Dropout(0.3)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT-BiLSTM Model Loaded")
        self.lstm2 = nn.LSTM(self.embeddim, self.hiddendim_lstm, batch_first=True)
        self.lstm1 = nn.LSTM(self.embeddim, self.hiddendim_lstm, batch_first=True)#, bidirectional=True) # noqa
        self.fc1 = nn.Linear(self.hiddendim_lstm * 2, self.numclasses)

    def attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)
        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1)
        #print(hidden_states.shape)
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)

        #print(last_hidden_state.shape)
        #print(hidden_states.shape)

        last_hidden_state = self.dropout(last_hidden_state)
        out1, _ = self.lstm1(last_hidden_state,None)
        out1 = self.dropout(out1[:, -1, :])


        #hidden_state = torch.randn(1 * 2, 1, self.hiddendim_lstm).to('cuda')  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        #cell_state = torch.randn(1 * 2, 1,  self.hiddendim_lstm).to('cuda')
        #out, (h_n, c_n) = self.lstm(hidden_states, (hidden_state,cell_state))
        hidden_states = self.dropout(hidden_states)
        out2, _ = self.lstm2(hidden_states, None)
        out2 = self.dropout(out2[:, -1, :])
        #out = out.permute(1, 0, 2)
        #query = self.dropout(out)
        #attn_output, attention = self.attention_net(out, query)
        #out2 = self.fc1(out2)
        #out = self.fc(attn_output)
        #print(out1.shape)
        #print(out2.shape)
        out = torch.cat((out2,out1),1)
        out = self.fc1(out)

        return out


class Bert_Attention(nn.Module):
    def __init__(self, numclasses, device):
        super(Bert_Attention, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT-att Model Loaded")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embeddim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embeddim, self.fchidden)) # noqa
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(device)

        self.fc = nn.Linear(self.fchidden, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class Bert_LSTM_Capsule(nn.Module):
    def __init__(self, numclasses):
        super(Bert_LSTM_Capsule, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.hiddendim_lstm = hiddendim_lstm
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT_LSTM_Capsule Model Loaded")
        self.lstm = nn.LSTM(self.embeddim, self.hiddendim_lstm, batch_first=True) # noqa
        self.fc = nn.Linear(self.hiddendim_lstm, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out, _ = self.lstm(hidden_states, None)

        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out