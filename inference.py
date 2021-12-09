import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    folder_list = os.listdir("test")
    folder_list.sort()
    test_list = []
    out_list = []
    for name in folder_list:
        if "mask" not in name and "predict" not in name:
            test_list.append("test/" + name)
            out_list.append("test/" + name.split(".")[0] + "_predict.png")
    
    in_files = test_list
    out_files = out_list

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- BIG ---
    net = UNet(n_channels=3, n_classes=2)
    net.to(device=device)
    net.load_state_dict(torch.load("checkpoints_big/checkpoint_epoch10.pth", map_location=device))
    
    # --- SMALL ---
    net_small = UNet(n_channels=3, n_classes=2)
    net_small.to(device=device)
    net_small.load_state_dict(torch.load("checkpoints_small/checkpoint_epoch5.pth", map_location=device))
    
    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)
                                   
        mask_small = predict_img(net=net_small,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask)
#         result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

        logging.info(f'Visualizing results for image {filename}, close to continue...')
        np_img = np.array(result)
        
        rate_list = []
        for y in range(480,581,10):
            check = 0
            count = 0
            for x in range(np_img.shape[1]):
                if check == 0 and np_img[y][x] == 127:
                    check = 1
                    count += 1
                if check == 1 and np_img[y][x] == 0:
                    check = 0
            rate_list.append(count)
        print("The number of heartbeats with Long lead II of " + out_filename + " : " + str(max(rate_list)) )
        
        
        result_small = mask_to_image(mask_small)
#         result_small.save(out_filename.split(".")[0] + "_small_predict.png")
    
        np_img_small = np.array(result_small)
        
        for y in range(np_img.shape[0]):
            for x in range(np_img.shape[1]):
                if np_img_small[y,x] == 127:
                    np_img[y,x] = 255
        
        Image.fromarray(np_img).save(out_filename)