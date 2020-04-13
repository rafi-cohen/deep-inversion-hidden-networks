import torch
import torch.nn as nn
import os
from torchvision import transforms

from parsing import parse_args
from deep_inversion import DeepInvert
from params import MODELS, LABELS


def evaluate(images, model, dataset, model_name, mean, std, cuda, *args, **kwargs):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    softmax = nn.Softmax(dim=1)
    original_model = MODELS[dataset][model_name](pretrained=True)
    original_model.eval()
    if cuda:
        original_model.cuda()
    original_preds = []
    original_confidences = []
    new_preds = []
    new_confidences = []
    for i, image in enumerate(images):
        image = preprocess(image).unsqueeze(0)
        if cuda:
            image = image.cuda()
        original_confidence, original_pred = torch.max(softmax(original_model(image)), dim=1)
        new_confidence, new_pred = torch.max(softmax(model(image)), dim=1)
        original_preds.append(original_pred)
        original_confidences.append(original_confidence)
        new_preds.append(new_pred)
        new_confidences.append(new_confidence)
    # convert lists to tensor
    original_preds = torch.stack(original_preds)
    original_confidences = torch.stack(original_confidences)
    new_preds = torch.stack(new_preds)
    new_confidences = torch.stack(new_confidences)
    # print stats
    unique_preds, counts = original_preds.unique(return_counts=True)
    print(*[f'(Pred: {pred}, Count: {count})' for pred, count in zip(unique_preds, counts)], sep=', ')
    print(f'Mean confidence = {original_confidences.mean()}')
    # verify that the model was not changed during training (i.e. all results are identical)
    assert (original_preds == new_preds).all()
    assert (original_confidences == new_confidences).all()


def main():
    args = parse_args()

    DI = DeepInvert(**vars(args))
    images = DI.deepInvert(**vars(args))
    for i, image in enumerate(images):
        label = LABELS[args.dataset][args.targets[i].item()]
        image.save(os.path.join(args.output_dir, f'{i}_{label}.png'))
    evaluate(images, **vars(args))


if __name__ == '__main__':
    main()
