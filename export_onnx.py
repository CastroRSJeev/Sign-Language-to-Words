import torch
import torch.nn as nn
import numpy as np

classes     = np.load("model/label_map.npy", allow_pickle=True)
num_classes = len(classes)

class LandmarkMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = LandmarkMLP(num_classes)
model.load_state_dict(torch.load("model/sign_model.pth", map_location="cpu"))
model.eval()

dummy = torch.zeros(1, 63)
torch.onnx.export(model, dummy, "model/sign_model.onnx",
                  input_names=["landmarks"], output_names=["logits"],
                  dynamic_axes={"landmarks": {0: "batch"}})
print("Exported to model/sign_model.onnx")
