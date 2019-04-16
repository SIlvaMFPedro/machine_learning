function idx = findClosestCentroids(X, centroids)
% idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids (idx)
%for a dataset X where each row is a single example. 
%idx is (m by 1) dimensional vector with integer values in the range [1..K])
%

% K -number of clusters
K = size(centroids, 1);

%number of examples
m=size(X,1);
% You need to return the following variables correctly.
idx = zeros(m, 1);

% ======= YOUR CODE HERE ======================
% Compute the Euclidian distance between every example i, 
%and every centroid and store in idx(i) the index of the closest centroid
% ind(i) should be a value in the range 1..K (the indexes of the clusters) 
%
%You can use a for-loop over the examples to compute this.
%
for i = 1:length(X)
    [~, ind] = min(sum((repmat(X(i,:),K,1) - centroids).^2, 2));
    idx(i) = ind;
    
end

% =============================================================
end

