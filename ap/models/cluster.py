import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .. import constants


class Cluster:

    EMPTY_CLUSTER_LABEL = -1

    def __init__(self, dataset, valid_dataset):

        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.unit_normalize = False
        self.z_normalize = False

        self.pca = None
        self.kmeans = None
        self.num_clusters = None
        self.cluster_vars = None

    def fit(self, num_clusters, seed=2019):

        self.num_clusters = num_clusters

        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=seed)
        self.kmeans.fit(self.dataset[constants.EMBEDDINGS])

        self.get_cluster_variances()

        return self.kmeans

    def get_cluster_variances(self):

        self.cluster_vars = []
        train_cluster_assignment = self.kmeans.predict(self.dataset[constants.EMBEDDINGS])

        for idx in range(self.num_clusters):

            points = self.dataset[constants.EMBEDDINGS][train_cluster_assignment == idx]
            self.cluster_vars.append(np.var(points, axis=0))

        self.cluster_vars = np.array(self.cluster_vars, dtype=np.float32)

    def transform_to_probability_space(self, embeddings):

        embeddings = self.preprocess_embeddings(embeddings)
        return self.get_mixture_probabilities(embeddings)

    def transform_to_hard_probabilities(self, embeddings):

        embeddings = self.preprocess_embeddings(embeddings)

        assignments = self.kmeans.predict(embeddings)
        one_hot = np.eye(self.num_clusters)[assignments]

        return one_hot.astype(np.float32)

    def transform_to_assignments(self, embeddings):

        embeddings = self.preprocess_embeddings(embeddings)
        return self.kmeans.predict(embeddings)

    def transform_continuous_transition_matrix(self, continuous_T, deterministic_transitions=False):

        centroids = self.kmeans.cluster_centers_

        if self.pca is not None:
            # transform centroids back to feature space
            centroids = self.pca.inverse_transform(centroids)

        discrete_T = np.empty((continuous_T.shape[0], len(centroids), len(centroids)), dtype=np.float32)

        transformed_centroids = np.matmul(centroids[:, np.newaxis, np.newaxis, :], continuous_T[np.newaxis, :, :, :])
        transformed_centroids = transformed_centroids[:, :, 0, :]

        if self.pca is not None:
            # transform transformed centroids from features space to PCA space
            original_shape = transformed_centroids.shape
            transformed_centroids = np.reshape(
                transformed_centroids, newshape=(original_shape[0] * original_shape[1], original_shape[2])
            )
            transformed_centroids = self.pca.transform(transformed_centroids)
            transformed_centroids = np.reshape(
                transformed_centroids, newshape=(original_shape[0], original_shape[1], transformed_centroids.shape[1])
            )

        for idx, transformed_centroid in enumerate(transformed_centroids):

            if deterministic_transitions:
                distances = self.kmeans.transform(transformed_centroid)
                assignment = np.argmin(distances, axis=1)
                probabilities = np.eye(distances.shape[1])[assignment]
            else:
                probabilities = self.get_mixture_probabilities(transformed_centroid)

            discrete_T[:, idx, :] = probabilities

        return discrete_T

    def transform_continuous_reward_matrix(self, continuous_R):

        centroids = self.kmeans.cluster_centers_

        if self.pca is not None:
            # transform centroids back to feature space
            centroids = self.pca.inverse_transform(centroids)

        discrete_R = np.matmul(centroids[np.newaxis, :, :], continuous_R[:, :, np.newaxis])[:, :, 0]

        return discrete_R

    def preprocess_embeddings(self, embeddings):

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        if self.unit_normalize:
            embeddings /= np.sum(np.square(embeddings), axis=1)[:, np.newaxis]

        if self.z_normalize:
            embeddings = (embeddings - self.means[np.newaxis, :]) / self.stds[np.newaxis, :]

        return embeddings

    def evaluate_purity(self, valid_cluster_assignment=None, blacklist=None):

        if valid_cluster_assignment is None:
            valid_cluster_assignment = self.kmeans.predict(self.valid_dataset[constants.EMBEDDINGS])

        cluster_purities = []
        cluster_sizes = []
        mean_purity = 0

        for idx in np.unique(valid_cluster_assignment):
            tmp_mask = valid_cluster_assignment == idx
            tmp_num_points = np.sum(tmp_mask)
            tmp_labels = self.valid_dataset[constants.STATE_LABELS][tmp_mask]

            tmp_values, tmp_counts = np.unique(tmp_labels, return_counts=True)
            tmp_purity = np.max(tmp_counts) / np.sum(tmp_counts)

            if blacklist is not None and tmp_values[np.argmax(tmp_counts)] in blacklist:
                tmp_purity = 0.0

            cluster_purities.append(tmp_purity)
            cluster_sizes.append(tmp_num_points)
            mean_purity += (tmp_num_points / len(self.valid_dataset[constants.STATE_LABELS])) * tmp_purity

        return cluster_purities, cluster_sizes, mean_purity

    def evaluate_purity_soft(self, cluster_probs, labels_key=constants.STATE_LABELS, mask=None):

        labels = self.valid_dataset[labels_key]
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

    def evaluate_inverse_purity_soft(self, cluster_probs, label_assignment, labels_key=constants.STATE_LABELS,
                                     mask=None):

        labels = self.valid_dataset[labels_key]
        if mask is not None:
            labels = labels[mask]

        scores = []
        totals = []

        for label in np.unique(labels):

            good = 0.0

            mask = labels == label

            mass = np.sum(cluster_probs[mask], axis=0)
            total = np.sum(mass)

            for cluster in range(self.num_clusters):

                if label_assignment[cluster] == label:
                    good += mass[cluster]

            score = good / total
            scores.append(score)
            totals.append(total)

        balanced_score = np.mean(scores)

        return balanced_score, scores, totals

    def evaluate_confusion(self, valid_cluster_assignment=None):

        if valid_cluster_assignment is None:
            valid_cluster_assignment = self.kmeans.predict(self.valid_dataset[constants.EMBEDDINGS])

        valid_labels = self.valid_dataset[constants.STATE_LABELS]
        num_labels = len(np.unique(valid_labels))

        matches = np.zeros((num_labels, num_labels))

        for i in range(num_labels):
            for j in range(num_labels):
                matches[i, j] = np.sum(np.logical_and(valid_cluster_assignment == i, valid_labels == j))

        correct = 0
        assigned_labels = {}

        for i in range(num_labels):
            max_coords = np.unravel_index(matches.argmax(), matches.shape)

            assigned_labels[max_coords[1]] = max_coords[0]

            correct += matches[max_coords]

            matches[max_coords[0], :] = -1
            matches[:, max_coords[1]] = -1

        return correct / valid_cluster_assignment.shape[0]

    def assign_label_to_each_cluster(self, cluster_assignment=None, validation=True):

        if validation:
            dataset = self.valid_dataset
        else:
            dataset = self.dataset

        if cluster_assignment is None:
            cluster_assignment = self.kmeans.predict(dataset[constants.EMBEDDINGS])

        labels = dataset[constants.STATE_LABELS]

        cluster_map = {}

        for cluster_idx in range(self.num_clusters):

            mask = cluster_assignment == cluster_idx
            values, counts = np.unique(labels[mask], return_counts=True)

            if len(values) == 0:
                cluster_map[cluster_idx] = self.EMPTY_CLUSTER_LABEL
            else:
                max_value = values[np.argmax(counts)]
                cluster_map[cluster_idx] = max_value

        return cluster_map

    def assign_label_to_each_cluster_soft(self, cluster_probs, validation=True, labels_key=constants.STATE_LABELS,
                                          mask=None):
        """
        Assign a label to each cluster using the distribution over clusters for each labeled point.
        :param cluster_probs:   NxK Tensor, where N is the number of samples and K is the number of clusters.
        :param validation:      Use validation data, otherwise use training data.
        :param labels_key:      Key of the labels in the dataset.
        :param mask:            Mask for the labels.
        :return:                Dictionary assigning each cluster a label.
        """

        if validation:
            dataset = self.valid_dataset
        else:
            dataset = self.dataset

        labels = dataset[labels_key]

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

        for cluster_idx in range(self.num_clusters):
            cluster_map[cluster_idx] = assignment[cluster_idx]

        return cluster_map

    def plot_embeddings_with_centroids(self, show=False):

        centroids = self.kmeans.cluster_centers_

        k_means_embeddings = np.concatenate([self.valid_dataset[constants.EMBEDDINGS], centroids], axis=0)
        k_means_labels = np.concatenate(
            [self.valid_dataset[constants.STATE_LABELS], [self.num_clusters] * self.num_clusters]
        )

        return model_utils.transform_and_plot_embeddings(k_means_embeddings, k_means_labels)

    def pca_to(self, num_components):

        self.pca = PCA(n_components=num_components)
        self.pca.fit(self.dataset[constants.EMBEDDINGS])

        self.dataset[constants.EMBEDDINGS] = self.pca.transform(self.dataset[constants.EMBEDDINGS])
        self.valid_dataset[constants.EMBEDDINGS] = self.pca.transform(self.valid_dataset[constants.EMBEDDINGS])

        return self.pca

    def normalize_sphere(self):

        self.unit_normalize = True

        self.dataset[constants.EMBEDDINGS] /= \
            np.sqrt(np.sum(np.square(self.dataset[constants.EMBEDDINGS]), axis=1)[:, np.newaxis])

        self.valid_dataset[constants.EMBEDDINGS] /= \
            np.sqrt(np.sum(np.square(self.valid_dataset[constants.EMBEDDINGS]), axis=1)[:, np.newaxis])

    def normalize_z(self):

        self.z_normalize = True

        self.means = np.mean(self.dataset[constants.EMBEDDINGS], axis=0)
        self.stds = np.std(self.dataset[constants.EMBEDDINGS], axis=0)

        self.dataset[constants.EMBEDDINGS] = (self.dataset[constants.EMBEDDINGS] -
            self.means[np.newaxis, :]) / self.stds[np.newaxis, :]

        self.valid_dataset[constants.EMBEDDINGS] = (self.valid_dataset[constants.EMBEDDINGS] -
            self.means[np.newaxis, :]) / self.stds[np.newaxis, :]

    def get_mixture_probabilities(self, x):

        centroids = self.kmeans.cluster_centers_
        num_centroids = centroids.shape[0]
        num_dimensions = centroids.shape[1]

        exp = - np.log(num_centroids) - (num_dimensions / 2) * np.log(2 * np.pi) - \
            (1 / 2) * np.sum(np.log(self.cluster_vars), axis=1)[np.newaxis, :] - \
            (1 / 2) * np.sum(((centroids[np.newaxis, :, :] - x[:, np.newaxis, :]) ** 2) / self.cluster_vars[np.newaxis, :, :], axis=2)

        return self.softmax(exp, axis=1)

    @staticmethod
    def softmax(x, axis):
        e_x = np.exp(x - np.max(x, axis=axis)[:, np.newaxis])
        return e_x / e_x.sum(axis=axis)[:, np.newaxis]

    @staticmethod
    def translate_cluster_labels_to_assigned_labels(cluster_labels, assigned_labels):

        translated_labels = []
        for cluster in cluster_labels:
            translated_labels.append(assigned_labels[cluster])

        return np.array(translated_labels)


class StateActionCluster:

    def __init__(self, state_cluster, action_clusters):

        self.state_cluster = state_cluster
        self.action_clusters = action_clusters

    def transform_continuous_transition_matrix(self, continuous_T, deterministic_transitions=False):

        discrete_T = np.empty((self.action_clusters[0].num_clusters, self.state_cluster.num_clusters,
                      self.state_cluster.num_clusters), dtype=np.float32)

        state_centroids = self.state_cluster.kmeans.cluster_centers_

        if self.state_cluster.pca is not None:
            state_centroids = self.state_cluster.pca.inverse_transform(state_centroids)

        transformed_centroids = np.matmul(
            state_centroids[:, np.newaxis, np.newaxis, :], continuous_T[np.newaxis, :, :, :]
        )
        transformed_centroids = transformed_centroids[:, :, 0, :]

        for c_idx in range(self.state_cluster.num_clusters):

            action_cluster = self.action_clusters[c_idx]
            action_centroids = action_cluster.kmeans.cluster_centers_

            if action_cluster.pca is not None:
                action_centroids = action_cluster.pca.inverse_transform(action_centroids)

            tmp_transformed_centroids = transformed_centroids[c_idx, :, :]
            tmp_transformed_centroids = \
                action_centroids[:, :, np.newaxis] * tmp_transformed_centroids[np.newaxis, :, :]
            tmp_transformed_centroids = np.sum(tmp_transformed_centroids, axis=1)

            if self.state_cluster.pca is not None:
                tmp_transformed_centroids = self.state_cluster.pca.transform(tmp_transformed_centroids)

            if deterministic_transitions:
                dists = self.state_cluster.kmeans.transform(tmp_transformed_centroids)
                assignment = np.argmin(dists, axis=1)
                probs = np.eye(dists.shape[1])[assignment]
            else:
                probs = self.state_cluster.get_mixture_probabilities(tmp_transformed_centroids)

            discrete_T[:, c_idx, :] = probs

        return discrete_T

    def transform_continuous_reward_matrix(self, continuous_R):

        discrete_R = np.empty((self.action_clusters[0].num_clusters, self.state_cluster.num_clusters), dtype=np.float32)

        abstract_states_R = self.state_cluster.transform_continuous_reward_matrix(continuous_R)

        for c_idx in range(self.state_cluster.num_clusters):

            action_cluster = self.action_clusters[c_idx]

            action_centroids = action_cluster.kmeans.cluster_centers_

            if action_cluster.pca is not None:
                action_centroids = action_cluster.pca.inverse_transform(action_centroids)

            tmp_R = np.matmul(action_centroids, abstract_states_R)[:, c_idx]

            discrete_R[:, c_idx] = tmp_R

        return discrete_R
