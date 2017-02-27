function images = ptychographyForwardTransform(im, N, numBlocks, overlap)
% im is the images
% N is the pixel size
% is the number of blocks to cut the image into
% overlap is the overlap between each block

% convert to grayscale
im = imresize(im, [N,N]);

% define parameters of block sampling
[opts.NY, opts.NX] = size(im);
opts.BlockSpacing = floor(N/(numBlocks-1));
opts.BlockW = opts.BlockSpacing*overlap;
opts.NBlocksX = numBlocks;
opts.NBlocksY = numBlocks;


% slicing variables
A = makeSlicingMatrixNoApp(opts); 
slicingOperatorForward = @(x)(A*x(:));
fftim = @(x)(fftshift(fft2(fftshift(x))));
ifftBlocks = @(x)(ifftshift(ifftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2));
reshapeBlocks = @(x)(reshape(x(:), [opts.BlockW opts.BlockW opts.NBlocksY opts.NBlocksX]));

images = abs(ifftBlocks(reshapeBlocks(slicingOperatorForward((im)))));
images = reshape(images, [opts.BlockW opts.BlockW opts.NBlocksY*opts.NBlocksX]);


