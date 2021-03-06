from __future__ import print_function
import random
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from PyTorch.loader import load_semeval, load_twitter, get_loader,load_selfdata
from PyTorch.Tokenizer import tokenize_sentences, get_pretrained_tokenizer
from PyTorch.train import train_model

SEED = 1345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    parser = argparse.ArgumentParser(description='BERT for ABSA')
    parser.add_argument('--dataset', type=str, default='self',
                        help='Dataset: laptop or restaurant (default: laptop)')
    parser.add_argument('--maxlen', type=int, default=100,
                        help='Maximum Sentence length for BERT')
    parser.add_argument('--numclasses', type=int, default=2,
                        help="to include conflict class for ABSA if yes: 4")
    parser.add_argument('--model_name', type=str, default='lstm',
                        help='model type')
    parser.add_argument('--data-path', type=str, default=None,
                        help='path to folder containing datasets')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch Size')
    parser.add_argument('--numepochs', type=int, default=15,
                        help='Number of epochs to train')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to report results')

    args = parser.parse_args()
    # Number of classes
    numclasses = args.numclasses
    # Dataset name
    dataset = args.dataset
    # Path to Dataset
    datapath = args.data_path

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #DEVICE = torch.device('cpu')
    if(args.dataset not in ['laptop', 'restaurant', 'twitter', 'self']):
        raise ValueError('Invalid dataset selected')
    if(args.model_name not in ['lstm', 'attention', 'base']):
        raise ValueError('Invalid model selected')

    if(dataset == 'laptop' or dataset == 'restaurant'):
        (train_sentence, train_aspect, train_sentiment,
         test_sentence, test_aspect, test_sentiment) = load_semeval(dataset,
                                                                    numclasses,
                                                                    datapath)
    elif(dataset == 'twitter'):
        (train_sentence, train_aspect, train_sentiment,
         test_sentence, test_aspect, test_sentiment) = load_twitter(dataset,
                                                                    datapath)
    elif (dataset == 'self'):
        (train_sentence, train_aspect, train_sentiment,
         test_sentence, test_aspect, test_sentiment) = load_selfdata()

    (train_sentence, dev_sentence, train_aspect, dev_aspect,
     train_sentiment, dev_sentiment) = train_test_split(train_sentence,
                                                        train_aspect,
                                                        train_sentiment,
                                                        test_size=0.1,
                                                        random_state=42)

    print("Training Data size: {}".format(len(train_sentence)))
    print("Validation Data size: {}".format(len(dev_sentence)))
    print("Test Data size: {}".format(len(test_sentence)))
    print("------------------------------------------------")
    # Returns Pretrained BERT Tokenizer
    bert_tokenizer = get_pretrained_tokenizer()

    print("Tokenizing training data")
    (train_input_ids, train_attention_masks,
     train_token_type_ids) = tokenize_sentences(bert_tokenizer,
                                                train_sentence,
                                                train_aspect,
                                                args.maxlen)
    train_labels = torch.from_numpy(np.asarray(train_sentiment, 'int32'))

    print("Tokenizing validation data")
    (dev_input_ids, dev_attention_masks,
     dev_token_type_ids) = tokenize_sentences(bert_tokenizer,
                                              dev_sentence,
                                              dev_aspect,
                                              args.maxlen)
    dev_labels = torch.from_numpy(np.asarray(dev_sentiment, 'int32'))

    print("Tokenizing test data")
    (test_input_ids, test_attention_masks,
     test_token_type_ids) = tokenize_sentences(bert_tokenizer,
                                               test_sentence,
                                               test_aspect,
                                               args.maxlen)
    test_labels = torch.from_numpy(np.asarray(test_sentiment, 'int32'))
    print("-------------------------------------------------")
    train_loader = get_loader(train_input_ids, train_attention_masks,
                              train_token_type_ids, train_labels,
                              args.batch_size)
    dev_loader = get_loader(dev_input_ids, dev_attention_masks,
                            dev_token_type_ids, dev_labels,
                            args.batch_size)
    test_loader = get_loader(test_input_ids, test_attention_masks,
                             test_token_type_ids, test_labels,
                             args.batch_size)

    train_model(train_loader, dev_loader, test_loader, args.model_name,
                numclasses, args.numepochs, args.runs, DEVICE)


if __name__ == '__main__':
    main()
