import albumentations as A


train_transform = A.Compose(
[
        A.Affine(scale=1, translate_percent=None, translate_px=None, rotate=0, shear={"x": (-10, 10), "y": (-0.1, 0.1)}, interpolation=1, mask_interpolation=0, fit_output=False, keep_ratio=False, rotate_method='largest_box', p=0.5),
        A.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
        A.GaussNoise(std_range=(0.1, 0.5), mean_range=(0, 0), per_channel=True, p=0.5),
        A.PixelDropout(dropout_prob=0.02, per_channel=False, drop_value=0, mask_drop_value=None, p=0.5),
        A.RandomGravel (gravel_roi=(0.2, 0.4, 0.9, 0.9), number_of_patches=2, p=0.5),
        A.AdvancedBlur (blur_limit=(3, 7), sigma_x_limit=(0.2, 1.0), sigma_y_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), p=0.5)    
    ]
)