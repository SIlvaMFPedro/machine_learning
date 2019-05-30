NumberOfImages = 5;     % choose the number of images you want to give as input.
prefix_image = 'C:\Users\silva\OneDrive - Universidade de Aveiro\AA\projecto\LabWork\atlascar_frames\01\frame';
file_format = '.jpg'

img = imread(strcat(prefix_image, num2str(0), file_format));
img = rgb2gray(img);
[r,c]= size(img);
X = NaN(r,c,1); %pre-allocate memory
X(:,:,1) = img;
% for k=2:NumberOfImages
%  X(:,:,k) = rgb2gray(imread(strcat(prefix_image, num2str(k), file_format)));
% end
save('frame_01.mat','X');

FileData = load('frame_01.mat');
csvwrite('frame_01.csv', FileData.X);

FileData = load('projdata.mat');
csvwrite('projdata.csv', FileData.X);