import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default='./flowers', help = 'Path to folder of images')
    parser.add_argument('--arch',type= str, default = 'vgg16', help = 'Model architecture')
    parser.add_argument('--hidden_units', type = int, default = 4096, help='Number of hidden units')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help='Choose a learning rate of the optimizer')
    parser.add_argument('--gpu', action = "store", default = 'gpu',  help = 'Chose between CPU or GPU (GPU recommended)')
    parser.add_argument('--epochs', type = int, default = 2, help='Choose the number of Epochs')
    parser.add_argument('--save_dir', type = str, default ='checkpoint.pth', help = 'Where to save the checkpoint file')

    return parser.parse_args()