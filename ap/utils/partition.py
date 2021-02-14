import numpy as np


def assign_label_to_each_cluster_soft(cluster_probs, labels, num_clusters, mask=None):

    if mask is not None:
        labels = labels[mask]

    label_masses = []

    for label in np.unique(labels):

        tmp_mask = labels == label

        if np.sum(tmp_mask) == 0:
            label_mass = np.zeros(cluster_probs.shape[1], dtype=np.float32)
        else:
            label_mass = np.sum(cluster_probs[tmp_mask], axis=0)

        label_masses.append(label_mass)

    label_masses = np.stack(label_masses)
    assignment = np.argmax(label_masses, axis=0)

    cluster_map = {}

    for cluster_idx in range(num_clusters):
        cluster_map[cluster_idx] = assignment[cluster_idx]

    return cluster_map


def evaluate_purity_soft(cluster_probs, labels, mask=None):

    if mask is not None:
        labels = labels[mask]

    label_masses = []

    for label in np.unique(labels):
        label_mass = np.sum(cluster_probs[labels == label], axis=0)
        label_masses.append(label_mass)

    label_masses = np.stack(label_masses)

    sizes = np.sum(cluster_probs, axis=0)
    sizes[sizes == 0.0] = 1.0
    purities = np.max(label_masses, axis=0) / sizes
    mean_purity = np.sum(purities * sizes) / np.sum(sizes)

    return purities, sizes, mean_purity


def evaluate_purity_balanced(cluster_probs, labels, mask=None):

    if mask is not None:
        labels = labels[mask]

    label_masses = []

    for label in np.unique(labels):
        label_mass = np.sum(cluster_probs[labels == label], axis=0)
        label_masses.append(label_mass)

    label_masses = np.stack(label_masses)

    sizes = np.sum(cluster_probs, axis=0)
    sizes[sizes == 0.0] = 1.0

    ids = np.argmax(label_masses, axis=0)
    label_purities = {}

    for label in np.unique(labels):
        majority_mask = ids == label
        minority_mask = np.bitwise_not(majority_mask)
        if np.sum(majority_mask) > 0:
            majority_mass = np.sum(label_masses[label, majority_mask])
            minority_mass = np.sum(label_masses[label, minority_mask])
            label_purity = majority_mass / (majority_mass + minority_mass)
            label_purities[label] = label_purity
        else:
            label_purities[label] = 0.0

    mean_purity = np.mean(list(label_purities.values()))

    return label_purities, sizes, mean_purity


def evaluate_inverse_purity_soft(cluster_probs, labels, label_assignment, num_clusters, mask=None):

    if mask is not None:
        labels = labels[mask]

    scores = []
    totals = []

    for label in np.unique(labels):

        good = 0.0

        mask = labels == label

        mass = np.sum(cluster_probs[mask], axis=0)
        total = np.sum(mass)

        for cluster in range(num_clusters):

            if label_assignment[cluster] == label:
                good += mass[cluster]

        score = good / total
        scores.append(score)
        totals.append(total)

    balanced_score = np.mean(scores)

    return balanced_score, scores, totals


def translate_cluster_labels_to_assigned_labels(cluster_labels, assigned_labels):

    translated_labels = []
    for cluster in cluster_labels:
        translated_labels.append(assigned_labels[cluster])

    return np.array(translated_labels)
