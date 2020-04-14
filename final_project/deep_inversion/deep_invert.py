import torch
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from transforms import Denormalize

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class DeepInvert:
    def __init__(self, model, mean, std, cuda, amp_mode, loss_fn, reg_fn, *args, **kwargs):
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
                                                        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
                                                        transforms.ToPILImage()])
        self.cuda = cuda
        self.use_amp = APEX_AVAILABLE and amp_mode != 'off'
        self.amp_mode = amp_mode

    @torch.no_grad()
    def clip(self, image_tensor):
        for c, (m, s) in enumerate(zip(self.transformMean, self.transformStd)):
            image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        return image_tensor

    @torch.no_grad()
    def toImages(self, input):
        return [self.transformPostprocess(image) for image in input]

    def deepInvert(self, batch, iterations, targets, lr, jitter=0, *args, **kwargs):
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        input = torch.stack(transformed_images)

        input.requires_grad_(True)
        if self.cuda:
            input = input.cuda()
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        if self.use_amp:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.amp_mode,
                                                   keep_batchnorm_fp32=True, loss_scale="dynamic")
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                # apply jitter
                dx, dy = torch.randint(-jitter, jitter+1, size=(2,)).tolist()
                input.data = input.roll(dx, -1).roll(dy, -2).data
                output = self.model(input)
                optimizer.zero_grad()
                loss = self.loss_fn(output, targets)
                if self.reg_fn:
                    loss = loss + self.reg_fn(input)
                if self.use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                # clip the image after every gradient step
                input.data = self.clip(input.data)
                # unjitter
                input.data = input.roll(-dx, -1).roll(-dy, -2).data

                desc_str = f'#{i}: total_loss = {loss.item()}'
                pbar.set_description(desc_str)
                pbar.update()
        return self.toImages(input.cpu())
