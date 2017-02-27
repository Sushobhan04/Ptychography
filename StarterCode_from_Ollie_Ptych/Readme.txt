Here is the function - you can call it for example using:

im = im2double(imread('lena.jpg'));
nrBlocks = 9; % 9
imgSize = 256; %256
images = ptychographyForwardTransform(im, imgSize,nrBlocks,1);

1) The first parameter is the image matrix, 
2) The second is the desired size of the input image - you can modify this to be whatever you want, but the images should be square 
3) The third is the number of blocks, I chose 9 (as in 9x9), but we could try 5 if this is too much 
4) The fourth parameter is the overlap factor between blocks. 1 means that the blocks butt right next to each other without overlapping (in the frequency domain). I'd like to try overlap factors of .5, 1, 2, 4

You can visualize the output with the following code:

[n, ~, nb, ~] = size(images);
ims = reshape(images, n,n,sqrt(nb),sqrt(nb));
ims = reshape(permute(ims, [1 3 2 4]), n*sqrt(nb), sqrt(nb)*n);
imagesc(abs(ims).^.25); colormap(gray); axis image; 

and it should look like the visualizeOutput.jpg figure attached.