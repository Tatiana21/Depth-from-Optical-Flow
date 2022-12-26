function [Dx Dy] = estimateOpticFlow_HS(ImgSeq, eta)

%%   Horn, B.K.P. and Schunck, B.G. (1981). Determining optical flow. 
%       Artificial Intelligence 17, 185-203.
%   The discretization and implementation follows:
%   Bruhn, A., Weickert, J. Kohlberger, T., and Schnörr, C. (2006). A 
%       multigrid platform for real-time motion computation with 
%       discontinuity preserving variational methods. International 
%       Journal of Computer Vision 70(3), 257-277.
%
%   Written by Dhruv Ilesh Shah and Tanya Choudhary
%   EE702: Computer Vision, IIT Bombay (Spring 2018)

W       = fspecial('gaussian',[9 9],2); % Gaussian smoothening kernel (denoising)
DiffX   = [1, -1]; % x-derivative kernel
DiffY   = DiffX'; % y-derivative kernel
DiffT   = reshape([-1 1],[1 1 2]); % t-derivative kernel
omega   = 1.7; % parameter for successive over relaxation (SoR)
iNum    = 20; % number of iterations
showFlow= 1; % visualize flow

% Check if the provided sequence contains at least two frames.
[yNum xNum tNum] = size(ImgSeq);
if tNum ~= 2, 
    error('MATLAB:frameErr', ['This method requires %d frames ',...
        'but %d frames were provided!'], 2, tNum);
end

% Compute the partial derivatives in x, y, and t.
ImgSeqDx = imfilter(ImgSeq, DiffX, 'same', 'replicate');
ImgSeqDy = imfilter(ImgSeq, DiffY, 'same', 'replicate');
ImgSeqDt = convn(ImgSeq, DiffT, 'valid');
% Select the spatial and temporal valid part of the partial derivatives.
ktNum    = size(DiffT,3);
ValidT   = (1+floor((ktNum-1)/2)) : (tNum-floor(ktNum/2));
ImgSeqDx = squeeze(ImgSeqDx(:,:,ValidT));
ImgSeqDy = squeeze(ImgSeqDy(:,:,ValidT));
tNum     = length(ValidT);
Dx       = squeeze(zeros(yNum, xNum, tNum));
Dy       = squeeze(zeros(yNum, xNum, tNum));

% Compute the coefficients A, B, C, D, E, and F of the structure tensor
%     / A B C \
% J = | B D E |
%     \ C E F /

A = imfilter(ImgSeqDx.*ImgSeqDx,W, 'same', 'replicate');
B = imfilter(ImgSeqDx.*ImgSeqDy,W, 'same', 'replicate');
C = imfilter(ImgSeqDx.*ImgSeqDt,W, 'same', 'replicate');
D = imfilter(ImgSeqDy.*ImgSeqDy,W, 'same', 'replicate');
E = imfilter(ImgSeqDy.*ImgSeqDt,W, 'same', 'replicate');

% Repeat boundary values by 1.
A   = repeatBoundary(A, 1);
B   = repeatBoundary(B, 1);
C   = repeatBoundary(C, 1);
D   = repeatBoundary(D, 1);
E   = repeatBoundary(E, 1);
Dx  = repeatBoundary(Dx, 1);
Dy  = repeatBoundary(Dy, 1);
yNum    = yNum + 2;
xNum    = xNum + 2;
IndexY  = 2 : (yNum-1);
IndexX  = 2 : (xNum-1);

if tNum==1,
    for iter = 1:iNum,
        for iy = IndexY,
            for ix = IndexX,
                Dx(iy,ix) = (1-omega) * Dx(iy,ix) ...
                    + omega * (Dx(iy-1,ix) + Dx(iy+1,ix)...
                              +Dx(iy,ix-1) + Dx(iy,ix+1)...
                              -1/eta*(B(iy,ix)*Dy(iy,ix) + C(iy,ix))) ...
                             /(1/eta*A(iy,ix) + 4);
                Dy(iy,ix) = (1-omega) * Dy(iy,ix) ...
                    + omega * (Dy(iy-1,ix) + Dy(iy+1,ix) ...
                              +Dy(iy,ix-1) + Dy(iy,ix+1) ...
                              -1/eta*(B(iy,ix)*Dx(iy,ix) + E(iy,ix))) ...
                             /(1/eta*D(iy,ix) + 4);
            end
        end
        % Dx = copyBoundary(Dx, kNum);
        % Dy = copyBoundary(Dy, kNum);
        % Use the faster, direct way to copy boundary values.
        Dx(1,:) = Dx(2,:);    Dx(yNum,:) = Dx(yNum-1,:);
        Dx(:,1) = Dx(:,2);    Dx(:,xNum) = Dx(:,xNum-1);
        Dy(1,:) = Dy(2,:);    Dy(yNum,:) = Dy(yNum-1,:);
        Dy(:,1) = Dy(:,2);    Dy(:,xNum) = Dy(:,xNum-1);
        
        % Optionally, display the estimated flow of the current iteration.
        if showFlow, plotFlow(Dx,Dy,iter,iNum); end
    end
end

Dx = eliminateBoundary(Dx, 1);
Dy = eliminateBoundary(Dy, 1);


function plotFlow(Dx, Dy, iter, iNum)
    cla;
    [yNum xNum] = size(Dx);
    [Y X]   = ndgrid(1:yNum, 1:xNum);
    sample  = ceil(yNum/45);
    IndexY  = 1:sample:yNum;
    IndexX  = 1:sample:xNum;
    scale   = sample*2;
    quiver(X(IndexY,IndexX),Y(IndexY,IndexX),...
           scale*Dx(IndexY,IndexX),scale*Dy(IndexY,IndexX),0,'-k');
    title(sprintf('Iteration %d of %d.',iter,iNum));
    axis ij equal; axis([-10 xNum+10 -10 yNum+10]);
    drawnow;
    