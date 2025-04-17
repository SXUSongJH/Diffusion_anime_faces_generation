from torchvision import transforms
import numpy as np

def hf_transforms(examples, channels, transform):
    img_key = list(examples.keys())[0]
    if channels == 1:
        examples['img_tensor'] = [transform(image.convert("L")) for image in examples[img_key]]
    elif channels == 3:
        examples['img_tensor'] = [transform(image.convert('RGB')) for image in examples[img_key]]
    else:
        raise ValueError("Invalid number of channels")
    del examples[img_key]
    return examples

def hf_preprocess(
    dataset,
    image_size,
    channels,
    exist_label = True,
    need_label = False
):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    def transform_fn(examples):
        return hf_transforms(examples, channels, transform)

    if exist_label and not need_label:
        transformed_dataset = dataset.with_transform(transform_fn).remove_columns('label')
    else:
        transformed_dataset = dataset.with_transform(transform_fn)
    
    return transformed_dataset

def reverse_transform(x_start):
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return transform(x_start.squeeze())
