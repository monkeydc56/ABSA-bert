from ast import literal_eval
import torch.utils.data as utils
import pandas as pd

def load_twitter(dataset, data_path=None):
    """Loads Twitter dataset

    Arguments:
        dataset (str): Name of the dataset
        data_path (str): Path to the dataset

    """
    if(data_path is None):
        data_path = '../Data/'
    train_file = data_path + dataset + '/train.raw'
    test_file = data_path + dataset + '/test.raw'

    train_sentence = []
    train_aspect = []
    train_sentiment = []

    with open(train_file, 'r', encoding='latin1') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        if(lines[i+2][:-1] == '-1'):
            lines[i+2] = '2'
        curind = lines[i].find('$T$')
        asp = lines[i+1][1]
        sen = lines[i][0: curind] + " " + asp + " " + lines[i][curind + 3: -1]
        train_sentence.append(sen)
        train_aspect.append(asp)
        train_sentiment.append(literal_eval(lines[i + 2]))

    test_sentence = []
    test_aspect = []
    test_sentiment = []

    with open(test_file, 'r', encoding='latin1') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        if(lines[i+2][:-1] == '-1'):
            lines[i+2] = '2'
        curind = lines[i].find('$T$')
        asp = lines[i+1][1]
        sen = lines[i][0: curind] + " " + asp + " " + lines[i][curind + 3: -1]
        test_sentence.append(sen)
        test_aspect.append(asp)
        test_sentiment.append(literal_eval(lines[i+2]))

    train_sen_len = [len(sentence.split()) for sentence in train_sentence]
    train_asp_len = [len(aspect.split()) for aspect in train_aspect]

    print("----------------------------------------")
    print("Maximum Training data Sentence Length: {}".format(
                                                       max(train_sen_len)))
    print("Maximum Training data Aspect Length: {}".format(max(train_asp_len)))
    print("----------------------------------------")

    return (train_sentence, train_aspect, train_sentiment, test_sentence,
            test_aspect, test_sentiment)


def load_semeval(dataset, num_classes, data_path=None):
    """ Loads SemEval 14 datasets

    Arguments:
        dataset (str): Name of the datsets
        num_classes (int): Number of classes to consider
        data_path (str): Path to the dataset

    """
    label = {'negative': 0,
             'positive': 1,
             'neutral': 2,
             'conflict': 3}

    if(data_path is None):
        data_path = '../Data/'
    train_file = data_path + 'atsa-' + dataset + '/atsa_train.json'
    test_file = data_path + 'atsa-' + dataset + '/atsa_test.json'

    temp = open(train_file, 'r', encoding='latin1').read()
    train = literal_eval(temp)
    train_sentence = []
    train_aspect = []
    train_sentiment = []
    for xml in train:
        if(xml['sentiment'] == 'conflict' and num_classes == 3):
            continue
        train_sentence.append(xml['sentence'])
        train_aspect.append(xml['aspect'])
        train_sentiment.append(label[xml['sentiment']])

    temp = open(test_file, 'r', encoding='latin1').read()
    test = literal_eval(temp)
    test_sentence = []
    test_aspect = []
    test_sentiment = []
    for xml in test:
        if(xml['sentiment'] == 'conflict' and num_classes == 3):
            continue
        test_sentence.append(xml['sentence'])
        test_aspect.append(xml['aspect'])
        test_sentiment.append(label[xml['sentiment']])

    train_sen_len = [len(sentence.split()) for sentence in train_sentence]
    train_asp_len = [len(aspect.split()) for aspect in train_aspect]

    print("----------------------------------------")
    print("Maximum Training data Sentence Length: {}".format(
                                                       max(train_sen_len)))
    print("Maximum Training data Aspect Length: {}".format(max(train_asp_len)))
    print("----------------------------------------")

    return (train_sentence, train_aspect, train_sentiment, test_sentence,
            test_aspect, test_sentiment)


def load_selfdata():
    train = pd.read_csv('./res_T/train_after.csv')#数据地址，要是改成工程代码的话可以放在外面，实在没时间了
    test = pd.read_csv('./res_T/test_after_R.csv')
    train_sentence = []
    train_aspect = []
    train_sentiment = []
    test_sentence = []
    test_aspect = []
    test_sentiment = []
    for i in range(train.shape[0]):
        #sentence = train['text'][i]
        #sentence = train['text'][i] + '[SEP]' + train['polarity'][i]#一答一问
        #sentence = train['text'][i] + '[SEP]' + train['question'][i]  # 一答一问
        #sentence = train['text'][i] + '[SEP]' + train['category'][i]  # 一答一问
        sentence = train['text'][i] + train['term'][i]  # 一答一问
        train_sentence.append(sentence)
        #train_aspect.append(train['category'][i])#第二个答案
        train_aspect.append(train['polarity'][i])
        #train_aspect.append(train['term'][i])  # 第二个答案
        #train_aspect.append(train['answer'][i])  # 第二个答案
        train_sentiment.append(train['label'][i])#判断对话是否成立
        #train_sentiment.append(train['polarity_id'][i])
    for i in range(test.shape[0]):
        #sentence = test['text'][i]
        #sentence = test['text'][i] + '[SEP]' + test['polarity'][i]
        #sentence = test['text'][i] + '[SEP]' + test['question'][i]
        #sentence = train['text'][i] + '[SEP]' + test['category'][i]  # 一答一问
        sentence = test['text'][i] + test['term'][i]  # 一答一问
        test_sentence.append(sentence)
        #test_aspect.append(test['category'][i])
        #test_aspect.append(test['term'][i])
        #test_aspect.append(test['answer'][i])
        test_aspect.append(test['polarity'][i])
        test_sentiment.append(test['label'][i])
        #test_sentiment.append(test['polarity_id'][i])
    return (train_sentence, train_aspect, train_sentiment, test_sentence,
            test_aspect, test_sentiment)

def get_loader(input_ids, attention_masks, token_type_ids, labels, batchsize):
    """ Converts input values into a dataloader

    Arguments:
    input_ids (Tensors): Sentences converted to input ids
    attention_masks (Tensors): Attention masks of the words in a sentence
    token_type_ids (Tensors): Token ids for pair of sentences
    labels (Tensors): Labels of the Dataset
    batchsize (int): Batch size to train

    """
    array = utils.TensorDataset(input_ids, attention_masks, token_type_ids,
                                labels)
    loader = utils.DataLoader(array, batch_size=batchsize)
    return loader
