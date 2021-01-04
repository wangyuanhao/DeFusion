function [predLabel, nmi, silhScore] = performKmeans(D, groundTruth, numCluster)
rng(1);
predLabel = kmeans(D, numCluster,'Replicates',10);
nmi = MutualInfo(groundTruth, predLabel);
silhScore = mean(silhouette(D, predLabel));
end