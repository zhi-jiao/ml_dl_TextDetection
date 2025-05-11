import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Model and training configuration')
    
    # Model parameters
    parser.add_argument('--output_dim', type=int, default=2, help='Output dimension')
    parser.add_argument('--vocab_size', type=int, default=1478 + 1, help='Vocabulary size (+1 for padding)')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--seq_length', type=int, default=256, help='Sequence length')
    parser.add_argument('--model', type=str, default='LSTM', help='models selection')
    # TextCNN specific parameters
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters for TextCNN')
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5], help='Convolution kernel sizes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    
    # File paths
    parser.add_argument('--train_path', type=str, default='./data/paraphrasing_data/sentence/train.csv', help='Training data path')
    parser.add_argument('--test_path', type=str, default='./data/paraphrasing_data/sentence/test.csv', help='Test data path')
    parser.add_argument('--val_path', type=str, default='./data/paraphrasing_data/sentence/val.csv', help='Validation data path')
    parser.add_argument('--dic_path', type=str, default='./data/paraphrasing_data/paraphrasing_data.pkl', help='Dictionary path')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    
    args = parser.parse_args()
    return args