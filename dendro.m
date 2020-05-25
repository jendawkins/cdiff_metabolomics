
% data = load('data.mat');
% targets = load('targets.mat');
% met_names = load('names.mat');

% X = data.all_data;
% T = clusterdata(X,cutoff);

NumCluster = 3;
rand('state', 7)
data = [rand(10,3); rand(10,3)+1; rand(10,3)+2]; 
dist = pdist(data, 'euclidean');
link = linkage(dist, 'complete');
clust = cluster(link, 'maxclust', NumCluster); 
[H,T,perm] = dendrogram(link, 0);