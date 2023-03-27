from sklearn.cluster import AgglomerativeClustering


k = [
    [70, 0, 134],
    [71, 0, 134],
    [70, 0, 133],
    [68, 0, 134],
    [30, 0, 145],
    [31, 0, 147],
    [79, 0, 147]
]

clustering = AgglomerativeClustering(
	n_clusters=2,
	affinity="manhattan",
	linkage="complete",
    )
clustering.fit(k)
print(k)
print(clustering.labels_)