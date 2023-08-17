import open_clip
import torch

def image_forward_fn(model, images, device, transform):
    images = torch.stack([transform(image) for image in images], dim=0).to(device) 
    return model.encode_image(images)

def text_forward_fn(model, texts, device, transform):
    texts = transform(texts).to(device)
    return model.encode_text(texts)

embedding_dim = 512

model, _, image_preprocess = open_clip.create_model_and_transforms(
        'xlm-roberta-base-ViT-B-32',
        pretrained='laion5b_s13b_b90k'
    )

text_preprocess = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
