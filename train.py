import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

# python3 train.py

num_points = 10
num_constraints = 5
point_dim = 2
num_epochs = 10000
lr = 0.1
check_period = 1
num_exp = 1  # 100


def cosine_sim(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def tuple_loss(p, q, r):
    # Input: Three points
    return cosine_sim(p - q, r - q)


def sort_PCA(points):
    mean = torch.mean(points, dim=0)
    centered_points = points - mean
    cov_matrix = torch.matmul(centered_points.T, centered_points) / (points.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    principal_axis = eigenvectors[:, -1]
    projections = torch.matmul(centered_points, principal_axis)
    sorted_indices = torch.argsort(projections)
    return sorted_indices.tolist(), principal_axis.tolist()


def score(constraints, sorted_indices):
    num_satisfied = 0
    for constraint in constraints:
        satiafied = False
        if sorted_indices.index(constraint[0]) < sorted_indices.index(constraint[1]) and sorted_indices.index(constraint[1]) < sorted_indices.index(constraint[2]):
            satiafied = True
        if sorted_indices.index(constraint[0]) > sorted_indices.index(constraint[1]) and sorted_indices.index(constraint[1]) > sorted_indices.index(constraint[2]):
            satiafied = True
        if satiafied:
            num_satisfied += 1
    return num_satisfied / len(constraints)


def plot(points):
    points_numpy = points.detach().numpy()
    plt.scatter(points_numpy[:, 0], points_numpy[:, 1], color="blue")
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('points.pdf')
    plt.savefig('points.png', dpi=600, bbox_inches='tight')
    plt.close()


for _ in range(num_exp):
    constraints = random.sample(list(itertools.combinations(range(num_points), 3)), num_constraints)

    points = torch.nn.Parameter(torch.randn(num_points, point_dim))
    optimizer = torch.optim.Adam([points], lr=lr)

    data = []
    best_score = -1
    best_epoch = None
    for epoch in range(num_epochs):
        loss = 0.0
        for constraint in constraints:
            loss += tuple_loss(points[constraint[0]], points[constraint[1]], points[constraint[2]])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % check_period == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
            sorted_indices, principal_axis = sort_PCA(points)
            # print('Axis :', principal_axis)
            # print('Order:', sorted_indices)
            curr_score = score(constraints, sorted_indices)
            # print('Score:', curr_score)
            if curr_score > best_score:
                best_score = curr_score
                best_epoch = epoch + 1
                data.append((best_epoch, best_score))
            plot(points)
            if best_score == 1:
                break
    print(data)
