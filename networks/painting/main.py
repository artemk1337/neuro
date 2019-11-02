import torch


model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model.eval()
