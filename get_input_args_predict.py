import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default='checkpoint.pth', help = 'Path to checkpoint')
    parser.add_argument('--category_names',type= str, default = 'cat_to_name.json', help = 'JSON file mapping categories to real names')
    parser.add_argument('--image_path', type = str, default = 'flowers/train/32/image_05611.jpg', help='Path to image for processing')
    parser.add_argument('--gpu', action = "store", default = 'gpu',  help = 'Chose between CPU or GPU (GPU recommended)')
    parser.add_argument('--topk', type = int, default = 5, help = 'Top K classes')

    return parser.parse_args()