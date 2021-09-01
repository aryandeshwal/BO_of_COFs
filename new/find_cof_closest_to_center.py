# find the COF closest to the center of the feature space.
x_center = 0.5 * np.ones(np.shape(X)[1])
id_closest_to_center = np.argmin(np.linalg.norm(X - x_center, axis=1))
X[id_closest_to_center, :]
