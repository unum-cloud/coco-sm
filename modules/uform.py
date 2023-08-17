import uform

def image_forward_fn(model, images, device, transform):
    images = model.preprocess_image(images).to(device)
    return model.encode_image(images)

def text_forward_fn(model, texts, device, transform):
    texts = model.preprocess_text(texts)
    texts = {k: v.to(device) for k, v in texts.items()}
    return model.encode_text(texts)

embedding_dim = 256
image_preprocess = None
text_preprocess = None

model = uform.get_model(
    'unum-cloud/uform-vl-multilingual-v2'
)

