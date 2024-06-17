import torch
import numpy as np


def binary_list(inval, length):
    """Convert x into a binary list of length l."""
    return np.array([int(i) for i in np.binary_repr(inval, length)])


def mask_flip(mask):
    """Interchange 0 <-> 1 in the mask."""
    return 1 - mask
def iflow_binary_masks(num_input_channel):
    """Create binary masks for to account for symmetries."""
    n_masks = int(np.ceil(np.log2(num_input_channel)))
    sub_masks = np.transpose(
        np.array([binary_list(i, n_masks) for i in range(num_input_channel)])
    )[::-1]
    flip_masks = mask_flip(sub_masks)

    # Combine masks
    masks = np.empty((2 * n_masks, num_input_channel))
    masks[0::2] = flip_masks
    masks[1::2] = sub_masks
    masks_new = [(torch.from_numpy(mask)).to(torch.uint8) for mask in masks]
    return masks_new

