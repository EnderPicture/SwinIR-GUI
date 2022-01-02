import os
import glob
import requests
import threading
import tkinter as tk
from tkinter import filedialog, Text
import cv2
import numpy as np
import torch
from models.network_swinir import SwinIR as net


def define_model(model_item):
    task = model_item['task']

    # 001 classical image sr
    if task == 'classical_sr':
        upscale = model_item['upscale']
        training_patch_size = model_item['training_patch_size']
        model = net(upscale=upscale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif task == 'lightweight_sr':
        upscale = model_item['upscale']
        model = net(upscale=upscale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif task == 'real_sr':
        model_size = model_item['model_size']
        if model_size == 'm':
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(model_item['path'])
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys(
    ) else pretrained_model, strict=True)

    return model


def setup(model_item):
    # 001 classical image sr/ 002 lightweight image sr
    if model_item['task'] in ['classical_sr', 'lightweight_sr']:
        border = model_item['scale']
        window_size = 8

    # 003 real-world image sr
    elif model_item['task'] in ['real_sr']:
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif model_item['task'] in ['gray_dn', 'color_dn']:
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif model_item['task'] in ['jpeg_car']:
        border = 0
        window_size = 7

    return border, window_size


def process(img_lq, model, model_item, window_size):

    tile_size = 400
    tile_overlap = 32

    if tile_size is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile_size, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        sf = model_item['upscale']

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run_model(model_item):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(model_item)
    model.eval()
    model = model.to(device)

    border, window_size = setup(model_item)

    for path in paths:
        (img_name, img_ext) = os.path.splitext(os.path.basename(path))

        print(img_name)

        image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        image = np.transpose(image if image.shape[2] == 1 else image[:, :, [
                             2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        image = torch.from_numpy(image).float().unsqueeze(
            0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = image.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            image = torch.cat([image, torch.flip(image, [2])], 2)[
                :, :, :h_old + h_pad, :]
            image = torch.cat([image, torch.flip(image, [3])], 3)[
                :, :, :, :w_old + w_pad]
            output = process(image, model, model_item, window_size)
            output = output[..., :h_old * model_item['upscale'], :w_old * model_item['upscale']]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            # CHW-RGB to HCW-BGR
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'output/{img_name}_SwinIR.png', output)


window = tk.Tk()
window.title('SwinIR')
window.minsize(width=500, height=100)

paths = []


def getPath():
    global paths
    paths = filedialog.askopenfilenames()


tk.Button(window, text='get file', command=getPath).pack()


MODLES = {
    'classicalSR s48 x2': {
        'task': 'classical_sr',
        'upscale': 2,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth',
    },
    'classicalSR s48 x3': {
        'task': 'classical_sr',
        'upscale': 3,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth',
    },
    'classicalSR s48 x4': {
        'task': 'classical_sr',
        'upscale': 4,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
    },
    'classicalSR s48 x8': {
        'task': 'classical_sr',
        'upscale': 8,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
    },
    'classicalSR s64 x2': {
        'task': 'classical_sr',
        'upscale': 2,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
    },
    'classicalSR s64 x3': {
        'task': 'classical_sr',
        'upscale': 3,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth',
    },
    'classicalSR s64 x4': {
        'task': 'classical_sr',
        'upscale': 4,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
    },
    'classicalSR s64 x8': {
        'task': 'classical_sr',
        'upscale': 8,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth',
    },
    'lightweightSR x2': {
        'task': 'lightweight_sr',
        'upscale': 2,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    },
    'lightweightSR x3': {
        'task': 'lightweight_sr',
        'upscale': 3,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth',
    },
    'lightweightSR x4': {
        'task': 'lightweight_sr',
        'upscale': 4,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    },
    'realSR M x4': {
        'task': 'real_sr',
        'model_size': 'm',
        'upscale': 4,
        'path': 'model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
    },
    'realSR L x4': {
        'task': 'real_sr',
        'model_size': 'l',
        'upscale': 4,
        'path': 'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
    },
    'gray denoise 15': {
        'task': 'gray_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth',
    },
    'gray denoise 25': {
        'task': 'gray_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth',
    },
    'gray denoise 50': {
        'task': 'gray_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    },
    'color denoise 15': {
        'task': 'color_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth',
    },
    'color denoise 25': {
        'task': 'color_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth',
    },
    'color denoise 50': {
        'task': 'color_dn',
        'upscale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    },
    'de-jpeg 10': {
        'task': 'jpeg_car',
        'upscale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth',
    },
    'de-jpeg 20': {
        'task': 'jpeg_car',
        'upscale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth',
    },
    'de-jpeg 30': {
        'task': 'jpeg_car',
        'upscale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth',
    },
    'de-jpeg 40': {
        'task': 'jpeg_car',
        'upscale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth',
    },
}
variable = tk.StringVar(window)
variable.set('choose one')  # default value
tk.OptionMenu(window, variable, *MODLES.keys()).pack()


def run():
    model_key = variable.get()
    if model_key in MODLES:
        model_item = MODLES[variable.get()]
        model_path = model_item['path']

        if os.path.exists(model_path):
            print(f'loading model from {model_path}')
            run_model(model_item)
        else:
            def download_model():
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{os.path.basename(model_path)}'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading model {model_path}')
                open(model_path, 'wb').write(r.content)
                run_model(model_item)
            threading.Thread(target=download_model).start()


tk.Button(window, text='run', command=run).pack()

window.mainloop()
