%% Depth from Optical Flow
%  This script attempts to estimate depth of a scene by estimating
%  the 2D optical flow. Ego (camera) motion is assumed to be
%  purely translational.

clear all;  close all;

% Load image sequence
addpath('./misc');
% filePattern = './../YosemiteWithClouds/ImgFrame%05d.pgm';
filePattern = './eval-data-gray/Army/frame%02d.png';
% filePattern = './frame%02d.png';
ImgSeq = readImgSeq(filePattern, 7, 8);


% Estimate optic flow for the image sequence.
eta = 0.08;
[Dx Dy] = estimateOpticFlow_HS(ImgSeq, eta);

% Display the estimated optic flow.
h       = size(ImgSeq,1);
w       = size(ImgSeq,2);
[Y X]   = ndgrid(1:h, 1:w); % pixel coordinates.
sample  = 5;
IndexX  = 1:sample:w;
IndexY  = 1:sample:h; 
len     = sample*2;
figure('Position',[50 50 600 600]); 
quiver(X(IndexY,IndexX),      Y(IndexY,IndexX),...
       Dx(IndexY,IndexX)*len, Dy(IndexY,IndexX)*len,0,'-k');
axis equal ij; axis([-10 w+10 -10 h+10]);
title('Estimated optical flow');

[z1, z2] = estimate_depth_from_flow(Dx, Dy);
imshow(mat2gray(-z1));
% Visualizing the generated depth maps
% figure('Position', [100, 100, 1200, 500]);
% subplot(1, 2, 1); imagesc(z1);
% title('Estimate using Dx');
% subplot(1, 2, 2); imagesc(z2);
% title('Estimate using Dy');