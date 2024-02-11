net = googlenet;
inputSize = net.Layers(1).InputSize(1:2);

%% 
img = zeros(height, width, 3, 'uint8');

%% 
width = inputSize(1);  % width in pixels
height = inputSize(2); % height in pixels

% Generate Gaussian noise for each color channel
noiseRed = randn(height, width);
noiseGreen = randn(height, width);
noiseBlue = randn(height, width);

% Scale and offset to get pixel values between 0 and 255
noiseImageRed = uint8(255 * mat2gray(noiseRed));
noiseImageGreen = uint8(255 * mat2gray(noiseGreen));
noiseImageBlue = uint8(255 * mat2gray(noiseBlue));

% Combine color channels
img = cat(3, noiseImageRed, noiseImageGreen, noiseImageBlue);
imshow(img)


%%

img = imread('dog.jpeg')
img = imresize(img, [224 224]);
imshow(img)

%%
imgDouble = im2double(img);
noisyImg = imnoise(imgDouble, 'gaussian', 0, 0.05);
imshow(noisyImg);
img = noisyImg;

%%

% Code section modified from: https://www.mathworks.com/help/deeplearning/ug/understand-network-predictions-using-occlusion.html
[YPred,scores] = classify(net,img);
[~,topIdx] = maxk(scores, 1);
imshow(img)
map = occlusionSensitivity(net,img,YPred);
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar
title(sprintf("Occlusion sensitivity (%s)", ...
    YPred))