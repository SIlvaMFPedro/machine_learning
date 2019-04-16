%% ML Class- K-Means Clustering
%
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%
%% Initialization
clear ; close all; clc

%%  Part 1: Find Closest Centroids ====================
% K-Means algorithm is divided into two phases: 
% findClosestCentroids and computeCentroids. 
%  Complete the code in findClosestCentroids function. 
%
fprintf('Finding closest centroids.\n\n');

% Load & plot dataset projdata.mat
load('projdata.mat')
% Select # of clusters and initial position of centroids
K = 3; % # of Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');


%% === Part 2: Compute Means ===================
%  After all examples were assigned to a cluster based on the closest centroid
%criteria, now you should complete computeCentroids function.
%
fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found before.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');


%% === Part 3: K-Means Clustering ================
%  After you have completed the functions computeCentroids and
%  findClosestCentroids, you can run the  kMeans algorithm. 
%
fprintf('\nRunning K-Means clustering on example dataset.\n\n');
% Settings for running K-Means
K = 3;
max_iters = 10;

% Here we set centroids to specific values
% but in practice you can generate them automatically, for example by
% settings them to randomly chosen data examples (as in kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells the function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

%% === Part 4: K-Means Clustering on Pixels ======
%  In this exercise, you will use K-Means to compress an image. To do this,
%  you will first run K-Means on the colors of the pixels in the image and
%  then you will map each pixel onto its closest centroid.
%  
%  You should now complete the code in kMeansInitCentroids.m
%

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load the image bird_small.png
% A = imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\práticas\LabWork7\images\bird_small.png')
% A = imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\práticas\LabWork7\images\black_white_bird.jpg')
% A = imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\práticas\LabWork7\images\color_specter.jpg')
% A = imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\práticas\LabWork7\images\ps.jpg')
% A = imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\práticas\LabWork7\images\test.jpg')

NumberOfImages = 1;     % choose the number of images you want to give as input.
prefix_image = 'C:\Users\silva\OneDrive - Universidade de Aveiro\AA\projecto\LabWork\frames\frame';
file_format = '.png'
template = rgb2gray(imread('C:\Users\silva\OneDrive - Universidade de Aveiro\AA\projecto\LabWork\templates\sinal.png'));
for n=1:NumberOfImages
    A = imread(strcat(prefix_image, num2str(n), file_format))
    % If imread does not work for you, you can try instead
    %   load ('bird_small.mat');

    %Extract each RGB dimensions. For example, the red color
    I=A;  I(:,:,2)=0;  I(:,:,3)=0; 
    subplot(221), imshow(I), title('Red color')

    A=double(A);%Transform into numerical format for computations
    A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

    % Size of the image
    img_size = size(A);

    % Reshape the image into an mx3 matrix where m = number of pixels.
    % Each row will contain the Red, Green and Blue pixel values
    % This gives the dataset matrix X used by K-Means.
    X = reshape(A, size(A, 1)*size(A, 2), 3);

    % Run K-Means algorithm on data X
    % You should try different values of K and max_iters here
    K = 16; 
    max_iters = 10;

    % When using K-Means, it is important the initialize the centroids
    % randomly. 
    % You should complete the code in kMeansInitCentroids.m
    initial_centroids = kMeansInitCentroids(X, K);

    % Run K-Means
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters);


    %% ====== Part 5: Image Compression ===============
    %  Now you will use the clusters of K-Means to compress an image. 
    %To do this, first find the closest clusters for each example. 

    fprintf('\nApplying K-Means to compress an image.\n\n');

    % Find closest cluster members
    idx = findClosestCentroids(X, centroids);

    % We can now recover the image from the indices (idx) by mapping each pixel
    % (specified by its index in idx) to the centroid value
    X_recovered = centroids(idx,:);

    % Reshape the recovered image into proper dimensions
    X_recovered = reshape(X_recovered, sqrt(size(X_recovered, 1)), sqrt(size(X_recovered, 1)), size(X_recovered, 2));

    % Display the original image 
    subplot(1, 2, 1);
    imagesc(A); 
    title('Original');

    % Display compressed image side by side
    subplot(1, 2, 2);
    imagesc(X_recovered)
    title(sprintf('Compressed, with %d colors.', K));
    
    %% ====== Part 6: Template Matching ===============
    
    % Load image data from compression result
    img1 = X_recovered;
    
    img = img1(:,:,1); 
    tmp = template(:,:,1);
    
    % Draw correlation map
    corr_map = zeros([size(img,1),size(img,2)]);
    
    for i = 1:size(img,1)-size(tmp,1)
        for j = 1:size(img,2)-size(tmp,2)
            %Construct the correlation map
            corr_map(i,j) = corr2(img(i:i+size(tmp,1)-1,j:j+size(tmp,2)-1),tmp);
        end
    end
    
    figure,imagesc(corr_map);colorbar;
    
    % Find the maximum value
    maxpt = max(corr_map(:));
    [x, y] = find(corr_map==maxpt);
    
    gray_img = rgb2gray(img1);
    Res   = img;
    Res(:,:,1)=gray_img;
    Res(:,:,2)=gray_img;
    Res(:,:,3)=gray_img;

    Res(x:x+size(tmp,1)-1,y:y+size(tmp,2)-1,:)=img1(x:x+size(tmp,1)-1,y:y+size(tmp,2)-1,:);
    
    % Display image result
    figure,imagesc(Res);
    
    
    % Load image data from compression result
    % img1 = rgb2gray(X_recovered);
    
    % Perform cross-correlation
    % c1 = normxcorr2(template,img1);  

    % Find peak in cross-correlation
    % [ypeak1, xpeak1] = find(c1==max(c1(:)));

    % Account for the padding that normxcorr2 adds
    % yoffSet1 = ypeak1-size(template,1);
    % xoffSet1 = xpeak1-size(template,2);
 
    % Displat matched area
    % hFig = figure;
    % hAx  = axes;
    % imshow(img1,'Parent', hAx);
    % imrect(hAx, [xoffSet1, yoffSet1, size(template,2), size(template,1)]);
    
    % Load image data from compression result
    % img2 = rgb2gray(X_recovered);
    
    % Perform cross-correlation
    % c2 = normxcorr2(template,img2);  

    % Find peak in cross-correlation
    % [ypeak2, xpeak2] = find(c2==max(c2(:)));

    % Account for the padding that normxcorr2 adds
    % yoffSet2 = ypeak2-size(template,1);
    % xoffSet2 = xpeak2-size(template,2);

    % Display matched area
    % hFig = figure;
    % hAx  = axes;
    % imshow(img2,'Parent', hAx);
    % imrect(hAx, [xoffSet2, yoffSet2, size(template,2), size(template,1)]);
    
    
    
    
    
end
