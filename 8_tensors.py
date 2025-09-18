import torch
import tensorflow as tf

'''
Rank 4 tensors in images, each dimention responds to:
1. Number of images in training batch
2. Image height in pixels
3. Image width in pixels
4. Number of colour channels. 3 for full-colour images (RGB)
'''

images_pt = torch.zeros([32, 28, 28, 3])

images_tf = tf.zeros([32, 28, 28, 3])

print(images_tf)