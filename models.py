MODLES = {
    'classicalSR s48 x2': {
        'task': 'classical_sr',
        'scale': 2,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth',
    },
    'classicalSR s48 x3': {
        'task': 'classical_sr',
        'scale': 3,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth',
    },
    'classicalSR s48 x4': {
        'task': 'classical_sr',
        'scale': 4,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
    },
    'classicalSR s48 x8': {
        'task': 'classical_sr',
        'scale': 8,
        'path': 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
    },
    'classicalSR s64 x2': {
        'task': 'classical_sr',
        'scale': 2,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
    },
    'classicalSR s64 x3': {
        'task': 'classical_sr',
        'scale': 3,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth',
    },
    'classicalSR s64 x4': {
        'task': 'classical_sr',
        'scale': 4,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
    },
    'classicalSR s64 x8': {
        'task': 'classical_sr',
        'scale': 8,
        'path': 'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth',
    },
    'lightweightSR x2': {
        'task': 'lightweight_sr',
        'scale': 2,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    },
    'lightweightSR x3': {
        'task': 'lightweight_sr',
        'scale': 3,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth',
    },
    'lightweightSR x4': {
        'task': 'lightweight_sr',
        'scale': 4,
        'path': 'model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    },
    'realSR M x4': {
        'task': 'real_sr',
        'model_size': 'm',
        'scale': 4,
        'path': 'model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
    },
    'realSR L x4': {
        'task': 'real_sr',
        'model_size': 'l',
        'scale': 4,
        'path': 'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
    },
    'gray denoise 15': {
        'task': 'gray_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth',
    },
    'gray denoise 25': {
        'task': 'gray_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth',
    },
    'gray denoise 50': {
        'task': 'gray_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    },
    'color denoise 15': {
        'task': 'color_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth',
    },
    'color denoise 25': {
        'task': 'color_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth',
    },
    'color denoise 50': {
        'task': 'color_dn',
        'scale': 1,
        'path': 'model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth',
    },
    'de-jpeg 10': {
        'task': 'jpeg_car',
        'scale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth',
    },
    'de-jpeg 20': {
        'task': 'jpeg_car',
        'scale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth',
    },
    'de-jpeg 30': {
        'task': 'jpeg_car',
        'scale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth',
    },
    'de-jpeg 40': {
        'task': 'jpeg_car',
        'scale': 1,
        'path': 'model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth',
    },
}
