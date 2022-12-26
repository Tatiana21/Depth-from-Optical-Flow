function [z1, z2] = estimate_depth_from_flow(Dx, Dy)

%% Estimate depth of a 2D scene given the 2D optical flow of the
%  scene. Ego (camera) motion is assumed to be purely translational.

U = Dx(:);
V = Dy(:);
x = 1:1:size(Dx,2);
X = repmat(x,size(Dx,1),1);
X = X(:);
y = 1:1:size(Dx,1);
Y = repmat(y',1,size(Dx,2));
Y = Y(:);

% Generating the G matrix
% 		/a d f\
% G = 	|d b e|
% 		\f e c/
a = (V')*V;
b = (U')*U;
c = ((X.*V - Y.*U)')*(X.*V - Y.*U);
d = -(V')*U;
e = (U')*(X.*V - Y.*U);
f = -(V')*(X.*V - Y.*U);

G = [a d f; d b e; f e c];

% Computing its eigen decomposition and extracting null space.
% or e-vec with smallest e-val
[V,D] = eig(G);
[~,min_ind] = min(diag(D));
t = V(:,min_ind);
u = t(1); v = t(2); w = t(3);
z1 = zeros(size(Dx));
z2 = z1;
for i = 1:size(Dx,1)
    for j = 1:size(Dx,2)
        z1(i,j) = (j*w - u)/Dx(i,j);
        z2(i,j) = (i*w - v)/Dy(i,j);
    end
end

thresh = 6;
z1(abs(z1) > thresh) = nan;
z2(abs(z2) > thresh) = nan;