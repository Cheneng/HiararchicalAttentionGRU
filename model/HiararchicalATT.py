import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from .BasicModule import BasicModule


class HiararchicalDoc(BasicModule):
    """
    Hiararchical attention GRU model for document classification.
    """
    def __init__(self, config):
        super(HiararchicalDoc, self).__init__()
        # Configuration of the model
        self.max_len = config.sentence_max_size
        self.input_dim = config.word_embedding_dimension
        self.hidden_dim = 50
        self.bidirectional = config.bidirectional
        self.drop_out_rate = config.drop_out
        self.context_vector_size = [100, 1]
        self.out_label_size = 2

        # dropout layer
        self.drop = nn.Dropout(p=self.drop_out_rate)

        # word level
        self.word_GRU = nn.GRU(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
        self.w_proj = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                out_features=2*self.hidden_dim)
        self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())
        self.softmax = nn.Softmax(dim=1)

        # sentence level
        self.sent_GRU = nn.GRU(input_size=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
        self.s_proj = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                out_features=2*self.hidden_dim)
        self.s_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())

        # document level
        self.doc_linear1 = nn.Linear(in_features=2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                     out_features=2*self.hidden_dim)
        self.doc_linear2 = nn.Linear(in_features=2*self.hidden_dim,
                                     out_features=self.hidden_dim)
        self.doc_linear_out = nn.Linear(in_features=self.hidden_dim,
                                        out_features=self.out_label_size)

    def forward(self, x, sent_num):
        # word level GRU
        x, _ = self.word_GRU(x)
        Hw = F.tanh(self.w_proj(x))
        w_score = self.softmax(Hw.matmul(self.w_context_vector))
        x = x.mul(w_score)
        x = torch.sum(x, dim=1)

        # sentence level GRU
        x = _align_sent(x, sent_num=sent_num).cuda()
        x, _ = self.sent_GRU(x)
        Hs = F.tanh(self.s_proj(x))
        s_score = self.softmax(Hs.matmul(self.s_context_vector))
        x = x.mul(s_score)
        x = torch.sum(x, dim=1)
        x = F.sigmoid(self.doc_linear1(x))
        x = F.sigmoid(self.doc_linear2(x))
        x = self.doc_linear_out(x)
        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels


def _align_sent(input_matrix, sent_num, sent_max=None):
    """
        To change presentation of the batch.

    Args:
        input_matrix (autograd.Variable):  The the presentation of the sentence in the passage. [sentence_num, sentence_embedding]
        sent_num (list): The list contains the number of sentences in each passage.
        max_len (int): The maximum sentence number of the passage.

    Returns:
        new_matrix (torch.FloatTensor): The aligned matrix, and its each row is one sentence in the passage.
                                        [passage_num, max_len, embedding_size]
    """
    # assert isinstance(input_matrix, torch.autograd.Variable), 'The input object must be Variable'

    embedding_size = input_matrix.shape[-1]      # To get the embedding size of the sentence
    passage_num = len(sent_num)                  # To get the number of the sentences
    if sent_max is not None:
        max_len = sent_max
    else:
        max_len = torch.max(sent_num)
    new_matrix = autograd.Variable(torch.zeros(passage_num, max_len, embedding_size))
    init_index = 0
    for index, length in enumerate(sent_num):
        end_index = init_index + length

        # temp_matrix
        temp_matrix = input_matrix[init_index:end_index, :]      # To get one passage sentence embedding.
        if temp_matrix.shape[0] > max_len:
            temp_matrix = temp_matrix[:max_len]
        new_matrix[index, -length:, :] = temp_matrix

        # update the init_index of the input matrix
        init_index = length
    return new_matrix