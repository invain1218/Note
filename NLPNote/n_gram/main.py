# main.py

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Tuple, Iterable, Optional

from functools import lru_cache
from collections import Counter, defaultdict

from utils import (
    generate_example,
    text_tokenize,
    load_arxiv_dataset,
    cross_validate,
    cross_validate_vocab,
)
from model import NGram
import vars

dir = os.path.dirname(os.path.abspath(__file__))
print(dir)


# 加载数据
load_start = time.time()

train_text, val_text = load_arxiv_dataset(vars.DATASETS_ROOT)

print(f"Dataset loaded in {time.time() - load_start:.2f}s")

print(train_text[:20])


# 参数
n_gram = 3
delta = 0.1
T = 0.5

alfas = [
    [0.05, 0.15, 0.8],
    [0.1, 0.3, 0.6],
    [0.33, 0.34, 0.33],
    [0.6, 0.3, 0.1],
    [0.8, 0.15, 0.05],
]

# 训练
model = NGram(
    train_text=train_text,
    n_gram=n_gram,
    delta=delta,
    alfas=alfas[0],
    min_c=3,
    min_C=3,
    T=T,
)

trained_model, best_alfas, ppl_map = cross_validate(
    model=model,
    val_text=val_text,
    alfas=alfas,
)

plt.figure(figsize=(10, 8))

# визуализация
ax = sns.barplot(
    x=list(map(str, ppl_map.keys())),
    y=list(ppl_map.values()),
)

min_idx = int(np.argmin(list(ppl_map.values())))
for i, bar in enumerate(ax.patches):
    if i == min_idx:
        bar.set_color("green")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f")

ax.set_title("Perplexity by alfas")
ax.set_xlabel("Alfas")
ax.set_ylabel("Perplexity")

plt.show()


# 调参
best_params, best_ppl, grid = cross_validate_vocab(
    train_text=train_text,
    val_text=val_text,
    n_gram=n_gram,
    delta=delta,
    alfas=best_alfas,
    min_c_list=[1, 3, 5],
    min_C_list=[1, 3, 5],
    T=T,
)

print(f"\nBEST: min_c={best_params[0]}, min_C={best_params[1]}")
print(f"Perplexity={best_ppl:.4f}")

df = pd.DataFrame(
    [
        {"min_c": k[0], "min_C": k[1], "perplexity": v}
        for k, v in grid.items()
    ]
)

pivot = df.pivot(index="min_c", columns="min_C", values="perplexity")

sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Perplexity vs vocabulary cleanup")
plt.show()


# MLE生成文本
best_model = NGram(
    train_text=train_text,
    n_gram=n_gram,
    delta=delta,
    alfas=best_alfas,
    min_c=best_params[0],
    min_C=best_params[1],
    T=T
)

check_text = "The phenomenon"
check_text_tokenized = text_tokenize(check_text)

generate_example(
    model=best_model,
    tokenized_text=check_text_tokenized,
    symbols_limit=100,
    stream=False
)