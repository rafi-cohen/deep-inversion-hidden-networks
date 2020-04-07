import torch
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from utils import Denormalize


class DeepInvert:
    def __init__(self, model, mean, std, cuda, loss_fn, reg_fn, *args, **kwargs):
        self.transformMean = mean
        self.transformStd = std

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.transformPreprocess = transforms.Normalize(mean=self.transformMean, std=self.transformStd)
        self.transformPostprocess = transforms.Compose([Denormalize(self.transformMean, self.transformStd),
                                                        transforms.Lambda(lambda x: x.clamp(0, 1) * 255),
                                                        transforms.ToPILImage()])
        self.cuda = cuda

    @torch.no_grad()
    def clip(self, image_tensor):
        for c, (m, s) in enumerate(zip(self.transformMean, self.transformStd)):
            image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        return image_tensor

    @torch.no_grad()
    def toImages(self, input):
        return [self.transformPostprocess(image) for image in input]

    def deepInvert(self, batch, iterations, target, lr, *args, **kwargs):
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        input = torch.stack(transformed_images)

        input.requires_grad_(True)
        if self.cuda:
            input = input.cuda()
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                output = self.model(input)
                optimizer.zero_grad()
                loss = self.loss_fn(output, target)
                if self.reg_fn:
                    loss = loss + self.reg_fn(input)
                loss.backward()
                optimizer.step()
                # clip the image after every gradient step
                input.data = self.clip(input.data)

                desc_str = f'#{i}: total_loss = {loss.item()}'
                pbar.set_description(desc_str)
                pbar.update()

        return self.toImages(input)
