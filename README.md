# DRAU-Net
1. install
- einops==0.6.1
- imageio==2.28.1
- albumentations
- Torchmetrics==0.11.4
  
2. datasets structure
- data
    - inria
        - train
             - image
             - label
        - val
             - image
             - label
        - test
             - image
             - label
3. train
-python train.py --dataset inria --n_timesteps 25

5. test
-python test.py --load_checkpoint output/checkpoints/20250401-0413_unet/20250401-0413_unet_e1.pt
