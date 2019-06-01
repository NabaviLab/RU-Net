function [outputArg1,outputArg2] = images_preprocessing(operation,imagesize)

clc
clear all
folder = uigetdir;
addpath(folder);
files = dir(fullfile(folder,'*.png')); 
files = {files.name};
operation='13';
if ~exist('operation=','var'), operation='6'; end     % defualt training folder
% choose operations on images:
% 1: sharp images
% 2: histogram equlisation
% 3: adaptive histogram equlisation
% 4: flip the images, upside down for odd images; left-right for even
% images
% 5: enhance colours
% 6: resize
% 7: make gray images into 3 channels
for i = 1:length(files)
    file = files{i};
    [FILEPATH,name,EXT] = fileparts(file);
    file=fullfile(folder,file);
   % file_temp=load(file);
    file_temp = imread(file);
    switch operation
        case '1'
            radius=10;
            amount=2;
            new_file = imsharpen(file_temp,'Radius',radius,'Amount',amount);
            radius_=num2str(radius);
            amount_=num2str(amount);
            filename=strcat(name,'.png');
            
        case '2'
            new_file = histeq(file_temp);
            filename=strcat('histeq_',name,'.png');
        case '3'
            file_temp=rgb2gray(file_temp) ;
            new_file =adapthisteq(file_temp,'NumTiles',[8 8]);
            new_file= cat(3,new_file,new_file,new_file);
            filename=strcat('adaphis_',name,'.png');
            
        case '4'
            new_file=fliplr(file_temp);
            filename = strcat(name,'.png');

        case '5'
            file_temp=rgb2gray(file_temp) ;
            new_file = adapthisteq(file_temp,'clipLimit',0.02,'Distribution','rayleigh');
            new_file= cat(3,new_file,new_file,new_file);
            filename = strcat('adapthisteq_',name,'.png');

        case '6'
            [height,width,channels]=size(file_temp);
            new_file=imresize(file_temp,imagesize);
            filename = strcat(name,'.png');


        case '7'
            new_file=cat(3,file_temp,file_temp,file_temp);
            filename = strcat(name,'.png');
            
    end
    fullname = fullfile(folder,filename);
    %figure
    %imshow(new_file)
    imwrite(new_file,fullname);
    fprintf('processing ...')
    fprintf('\n\n');    
end

disp('end')
end