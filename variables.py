import torch

epochs = 3
batch_size = 1
img_size = 512
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

train_loss_track = []
test_loss_track = []
test_iou_track = []