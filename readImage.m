function [] =readImage(imagesize)
files=dir('./*.png');  %change if jpg
% count=0
for file =files'
    image=imread(file.name);
    
    % image=cat(3,image,image,image);
    % imshow(image)
    % image=(image(:,:,1);
    
    imwrite(imresize(image,imagesize),file.name);
end
end