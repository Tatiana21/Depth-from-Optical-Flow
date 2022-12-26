function [u, v] = estimateOpticFlow_LK(images, window_size)

%%   A modern implementation of the idea of 
%    Lucas, B.D. and Kanade, T. (1981). An iterative image registration 
%       technique with and application to stereo vision. In Proceedings 
%       of Imaging Understanding Workshop, 121-130.
%
%   Written by Dhruv Ilesh Shah and Tanya Choudhary
%   EE702: Computer Vision, IIT Bombay (Spring 2018)

if size(images{1}, 3) == 3
    images{1} = rgb2gray(images{1});
    images{2} = rgb2gray(images{2});
end

images{1} = im2double(images{1});
images{2} = im2double(images{2});
% Calculate derivates
fx = conv2(images{1}, 0.25 * [-1 1; -1 1]) + conv2(images{2}, 0.25 * [-1 1; -1 1]);
fy = conv2(images{1}, 0.25 * [-1 -1; 1 1]) + conv2(images{2}, 0.25 * [-1 -1; 1 1]);
ft = conv2(images{1}, 0.25 * ones(2)) + conv2(images{2}, -0.25 * ones(2));

% Calculate optical flow
window_center = floor(window_size / 2);
image_size = size(images{1});
u = zeros(image_size);
v = zeros(image_size);
for i = window_center + 1:image_size(1) - window_center
  for j = window_center + 1:image_size(2) - window_center
    % Get values for current window
    fx_window = fx(i - window_center:i + window_center, j - window_center:j + window_center);
    fy_window = fy(i - window_center:i + window_center, j - window_center:j + window_center);
    ft_window = ft(i - window_center:i + window_center, j - window_center:j + window_center);

    fx_window = fx_window';
    fy_window = fy_window';
    ft_window = ft_window';

    A = [fx_window(:) fy_window(:)];

    U = pinv(A' * A) * A' * -ft_window(:);

    u(i, j) = U(1);
    v(i, j) = U(2);
  end
end

% Display the result
figure
axis equal
quiver(impyramid(impyramid(medfilt2(flipud(u), [5 5]), 'reduce'), 'reduce'), -impyramid(impyramid(medfilt2(flipud(v), [5 5]), 'reduce'), 'reduce'));