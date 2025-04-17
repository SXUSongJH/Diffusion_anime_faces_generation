import os
import matplotlib.pyplot as plt

def subplots_num(images_num, cols):
    if images_num <= cols:
        return (1, images_num)
    else:
        row_num = images_num // cols
        left_images = images_num % cols
        if left_images == 0:
            return (row_num, cols)
        else:
            return (row_num + 1, cols)

def show_images(
    idx,
    hf_datasets,
    image_size,
    cols_num=8,
    dpi = 300,
    save_dir='DDPM-Pytorch/results',
    save_name='image_examples'
):
    data = hf_datasets[idx]
    img_key = list(hf_datasets[0].keys())[0]
    images = data[img_key]
    rows, cols = subplots_num(len(images), cols_num)
    fig, axs = plt.subplots(rows, cols, figsize=(8*cols, 8*rows), dpi=dpi)
    axs = axs.flatten()
    for i, ax in enumerate(axs[:len(images)]):
        img = images[i]
        ax.axis('off')
        ax.imshow(img, interpolation='nearest')
    for ax in axs[len(images):]:
        ax.axis('off')
    plt.tight_layout()

    save_dir = os.path.join('..', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name + '.png')
    print(f'Saving images to: {save_path}')
    plt.savefig(save_path, bbox_inches='tight')
    if os.path.exists(save_path):
        print(f'Images successfully saved to: {save_path}')
    else:
        print(f'Failed to save images to: {save_path}')
    plt.close(fig)