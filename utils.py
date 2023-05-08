import torch
import numpy as np
import random
import matplotlib.pyplot as plt



def set_seed(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



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



def show_predictions(dataloader, model, device):
    model.train()
    fig, axs = plt.subplots(2, 3)
    [axi.set_axis_off() for axi in axs.ravel()]
    with torch.no_grad():
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        real_images = x.cpu()
        ground_truth = y.cpu()
        pred_mask = model(x).cpu()
        # Plot the real and predicted images in each subplot
        for i in range(2):
            for j in range(3):
                if j == 0:
                    axs[i, j].imshow(real_images[i][0])
                    axs[i, j].set_title("Real Image")
                elif j == 1:
                    axs[i, j].imshow(pred_mask[i][0]>pred_mask[i][0].mean())
                    axs[i, j].set_title("Predicted Mask")
                else:
                    axs[i, j].imshow(ground_truth[i][0])
                    axs[i, j].set_title("Ground Truth")
        plt.show()