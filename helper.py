import torch


def my_fn_align_sent(batch):
    # The return is [(data, labels, sent_num), (data, labels, sent_num)]
    batch_size = len(batch)
    data_batch = []
    labels_batch = []
    sent_num = []
    for i in range(batch_size):
        minibatch = batch[i]
        data_batch.extend(minibatch[0])
        labels_batch.extend(minibatch[1])
        sent_num.extend(minibatch[2])
    return torch.FloatTensor(data_batch), torch.LongTensor(labels_batch), torch.LongTensor(sent_num)
