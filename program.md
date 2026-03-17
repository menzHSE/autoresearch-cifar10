# autoresearch cifar-10

This is an experiment to have the LLM do its own research and improve a Resnet20 baseline from 2016 up to date.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`) and the time budget in prepare.py. The branch `autoresearch/<tag>` must not already exist, check for that.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master
3. **Read the in-scope files**: The repo is small. Read the files for full context:
   - `prepare.py`: fixed constants, and evaluation function. Do not modify
   - `train.py`: the file you modify. Model architecture, optimizer, lr, augmentation, regularization, training loop etc...
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget defined in prepare.py** (wall clock training time, excluding startup/compilation adn validation run). You launch it simply with: `uv run train.py`.


**What you CAN do:**
- Modify `train.py` this the only file you edit. Everything is faire game: model architecture, optimizer, data augmentation, hyperparameters, training loop, batch size, model size, model type etc...

**What you CANNOT do**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and time budget
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`
- Modify the evaluation harness. The `Eval.evaluate()` method in `prepare.py` is the ground truth metric.*


**The goal is simple: get the highest test accuracy (best_test_acc).** Since the training time budget is fixed and the validation time is removed focus only on getting the best hyperparameters and training code setup. The first constraint is that the code runs without crashing and finishes within the time budget. The second is not to run the validation more than once per epoch

**VRAM** is a soft constraint. Some increase if acceptable for meaningful increase in best_test_acc with this dataset you should have some leeway.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. The same accuracy from deleting code? Definitely keep. The same accuracy but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.


## Output format

Once the script finishes it prints a summary like this:

```
---
best_test_acc:    91.86%
final_test_acc:   91.86%
final_test_loss:  0.2543
training_seconds: 300.1
total_seconds:    325.9
startup_seconds:  3.2
peak_vram_mb:     1234.5
num_epochs:       164
num_steps:        64000
num_params:       272,474
```

You can extract the key metrics from the log file:

```
grep "^best_test_acc:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	best_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. best_test_acc achieved (e.g. 91.86) — use 0.00 for crashes
3. peak memory in GB, round to .1f (e.g. 1.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	best_acc	memory_gb	status	description
a1b2c3d	91.86	1.2	keep	baseline ResNet-20
b2c3d4e	92.34	1.3	keep	increase LR to 0.2
d4e5f6g	0.00	0.0	crash	double model width (OOM)
```


## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/<tag>`) that you defined previously and do the following pseudo code.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7.  Append the result to results.tsv using the Edit tool (add one line, never rewrite the whole file) (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If best_test_acc improved (higher), you "advance" the branch, keeping the git commit
9. If best_test_acc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take a few minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**pre-commit**: Ruff is used in pre-commit to have a consistent writing style. It will generally auto fix the issue but implement what you want do not feel limited by it. 

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?" or finish at "well-optimized" the purpose is to go to the limit. The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
