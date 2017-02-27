
function A = makeSlicingMatrixNoApp(opts)
% Create a sparse sampling matrix to sample overlapping
% patches in the fourier domain for ptychography
%
% inputs:
%
% opts: the paramters of the plenoptic camera
%   NX/NY: the pixels width of the sensor
%   BlockW: the width in pixels of each block
%   NBlocksX/Y: the number of blocks in 1D
%   BlockSpacing: the spacing between adjacent blocks
%
% output:
%
% A: the sparse sampling matrix

% create a full aperture
aper = ones(opts.BlockW,opts.BlockW);

% NX = opts.NX + (opts.BlockW - opts.apDia);
% NY = opts.NY + (opts.BlockW - opts.apDia);
NX = opts.NX + opts.BlockW;
NY = opts.NY + opts.BlockW;
NBlocksY = opts.NBlocksY;
NBlocksX = opts.NBlocksX;
BlockW = opts.BlockW;
spacing = opts.BlockSpacing;

k = 1;
rowInds = 1:BlockW^2*NBlocksX*NBlocksY';
colInds = zeros(BlockW^2*NBlocksX*NBlocksY,1);
apVals = zeros(BlockW^2*NBlocksX*NBlocksY, 1);

for i = 1:opts.NBlocksY
    for j = 1:opts.NBlocksX
        y1 = (i-1)*spacing+1;
        x1 = (j-1)*spacing+1;
        iy = floor(y1):floor(y1)+BlockW-1;
        ix = floor(x1):floor(x1)+BlockW-1;
        ap = imtranslate(aper, [y1-floor(y1) x1-floor(x1)]);
        
        % create a 2D grid of the coordinates in the aperture plane for
        % this slice - these are the column indices
        [IY, IX] = meshgrid(iy,ix);
        
        colInds(((k-1)*BlockW^2+1):k*BlockW^2) = sub2ind([NY,NX],IX(:),IY(:));
        apVals(((k-1)*BlockW^2+1):k*BlockW^2) = ap(:);
        
        k = k+1;
    end
end
% populate the sparse matrix
A = sparse(rowInds, colInds, apVals, BlockW^2*NBlocksX*NBlocksY,NX*NY);

% just grab the columns that we care about

mask = padarray(ones(opts.NY,opts.NX), round(opts.BlockW/2)*[1 1]); %even block sizes will round to the lower number, odd block sizes will round larger
mask = mask(1:NY,1:NX); %crop mask in case of odd block size

A = A(:,mask(:)==1);

