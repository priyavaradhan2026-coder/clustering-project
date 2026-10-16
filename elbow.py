from kneed import KneeLocator
from sklearn.cluster import KMeans

def get_optimum_k(data):
    #i)find wcss
    wcss = []
    k_values=range(1,11)
    for  i in k_values:
        model = KMeans(n_clusters=i)
        model.fit(data)
        wcss.append(model.inertia_)
    #ii)optimum K
    KL = KneeLocator(k_values,wcss,
                     curve="convex",
                     direction="decreasing")
    return KL.elbow