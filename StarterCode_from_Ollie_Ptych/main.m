output_path = 'D:\Documents\Research_data\ptychography\dataset_1.h5';
source = 'D:\Documents\Research_data\fourier_ptychography\Set91\';

files = dir(strcat(source,'*.bmp'));
nrBlocks = 3; % 9
imgSize = 256; %256
N = imgSize/(nrBlocks-1);

b = fix((imgSize-N)/2)+1;
e = fix((imgSize+N)/2);

dataset = zeros(N,N,nrBlocks^2,size(files,1));
label = zeros(N,N,1,size(files,1));
i=0;


for file = files'
    i = i+1;
%     disp('here');
    im = strcat(source,file.name); 
    im = imresize(rgb2gray(im2double(imread(im))),[imgSize,imgSize]);
    set = ptychographyForwardTransform(im, imgSize,nrBlocks,1);
    set = set./max(max(set));
    dataset(:,:,:,i) = set;
    label(:,:,:,i) = im(b:e,b:e,:);
    
    
end



fprintf('creating db');
h5create(output_path,'/data',size(dataset),'Datatype','single');
h5create(output_path,'/label',size(label),'Datatype','single');
h5write(output_path,'/data',dataset);
h5write(output_path,'/label',label);
fprintf('created db');

disp(size(dataset));
disp(size(label));