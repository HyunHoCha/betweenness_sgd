import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random


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
