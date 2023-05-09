import torch
import random
import numpy as np
from variables import *
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex


def set_seed(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss tracking
        train_loss_track.append(loss.item())

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.train()
    # IOU score i.e intersetion over union, ranges between 0-1, the higher the better
    test_loss, iou_score = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            jaccard = JaccardIndex(task="binary").to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            iou = jaccard(pred,y)
            iou_score += iou.item()
            test_loss += loss.item()
            # loss tracking
            test_loss_track.append(loss.item())
            test_iou_track.append(iou.item())

    iou_score /= num_batches
    test_loss /= num_batches
    print(f"Test loss: {test_loss:.4f}, IOU score: {iou_score:.4f}")



def visualize_imgs(dataloader, is_mask=False):
    a = 1 if is_mask else 0
    fig = plt.figure(figsize=(5, 5))
    columns = 2
    rows = 1
    for i in range(1, columns*rows +1):
        img = next(iter(dataloader))[a][i-1]
        img = torch.transpose(img, 0, 2)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis("off")
    plt.show()



def show_predictions(dataloader, model):
    model.train()
    fig, axs = plt.subplots(1, 3)
    [axi.set_axis_off() for axi in axs.ravel()]
    with torch.no_grad():
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        real_images = x.cpu()
        ground_truth = y.cpu()
        pred_mask = model(x).cpu()
        # Plot the real and predicted images in each subplot
        for i in range(3):
            if i == 0:
                axs[i].imshow(real_images[i][0])
                axs[i].set_title("Real Image")
            elif i == 1:
                axs[i].imshow(pred_mask[i][0]>pred_mask[i][0].mean())
                axs[i].set_title("Predicted Mask")
            else:
                axs[i].imshow(ground_truth[i][0])
                axs[i].set_title("Ground Truth")
        plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)