# autoresearch-cifar-10

An adaptation of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) repository applied on CIFAR-10 classification.
Starting from the Resnet paper I made the code as close as possible to the Resnet-20 setup and let Claude Opus 4.6 apply more modern techniques to boost performance. 
And yes CIFAR-10 makes it possible to run on any GPU and not just H100


## Setup

```bash
# Clone the repository
git clone https://github.com/GuillaumeErhard/autoresearch-cifar10.git
cd autoresearch-cifar10

uv sync
uv run pre-commit install
```

Choose your training time budget in prepare.py and commit the file. Make sure you have no previous result file in the folder. Then run your agent like Claude/Codex with a simple prompt like

```
Hi have a look at program.md and let's kick off a new experiment!
```

## Baseline: ResNet-20

The baseline model is a faithful implementation of the CIFAR-10 ResNet-20 from [He et al. 2015](https://arxiv.org/abs/1512.03385) (Section 4.2). The architecture, training schedule, and hyperparameters match the paper:

- 6n+2 = 20 layers (n=3), with filter widths {16, 32, 64}
- Identity shortcuts with zero-padding
- SGD with LR 0.1, momentum 0.9, weight decay 1e-4
- LR divided by 10 at 32k and 48k iterations, trained for 64k iterations
- Batch size 128, Kaiming initialization, no dropout
- Data augmentation: 4px padding + random 32x32 crop + horizontal flip

One minor deviation from the paper, following standard practice:

1. **Channel-wise mean** instead of per-pixel mean subtraction. The paper specifies a 32x32x3 mean image, but per-pixel mean doesn't align spatially after random cropping, making it semantically questionable under augmentation. Channel-wise mean is what later codebases (including the authors') converged on. But I decided to stick to per channel as it needs to be constant in eval.

Running this code gives a 91.89 % accuracy a bit better than the stated 91.25 %. But yes looking at this implementation there is a lot to squeeze with a more modern approach.
