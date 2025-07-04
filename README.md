# Continuous Optimization for Solving the Betweenness Problem

This program proposes a novel approach to solving the **betweenness problem**, a well-known combinatorial optimization task, by leveraging **continuous optimization techniques** instead of discrete methods.

---

## Features

- **Loss Function Design**:
  - Uses a custom loss function that includes cosine similarity to encode the betweenness constraints.

- **Solving NP-Complete Problems Practically**:
  - Offers practical heuristic solutions to problems that are NP-complete.

---

## Main Functions

- **Element-to-Point Mapping**:
  - Maps elements to points in continuous space and finds layouts that satisfy betweenness constraints.

- **Applicability**:
  - Useful in domains like **bioinformatics**, **combinatorial optimization**, and other structured ordering tasks.

---

## How to Use

```bash
python3 train.py  # Run training to solve betweenness problem
