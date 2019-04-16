function centroids = computeCentroids(X, idx, K)
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new cluster centroids 
%by computing the means of the data points assigned to each cluster. 
%X  - data matrix where each row is a single example
%idx - vector of cluster assignments (values in the range [1..K]) for each example
%K - number of centroids. 
%   centroids - each row of centroids is the mean of data points assigned to it.
%
% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);
% ====YOUR CODE HERE ======================
% Go over every centroid and compute the mean of all examples (points) that
% belong to it. Concretely, the row vector centroids(i, :)
% should contain the mean of the data points assigned to centroid i.
% You can use a for-loop over the centroids to compute this.
%
for i=1:K
    indices = find(idx==i);
    centroids(i, :) = mean(X(indices, :));
end

% =============================================================
end
