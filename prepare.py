import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 8
TIME_BUDGET_S = 300
DATASET_DIR = "./data"


class Eval:
    def __init__(self):
        mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)

        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_set = datasets.CIFAR10(
            DATASET_DIR, train=False, download=True, transform=test_tf
        )
        self.loader = DataLoader(
            test_set,
            batch_size=256,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    @torch.inference_mode()
    def evaluate(self, model, device):
        model.eval()

        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in self.loader:
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        return total_loss / total, 100.0 * correct / total
