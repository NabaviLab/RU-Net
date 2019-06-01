function [metrics,normConfMatData] = Test_semantic_segmentation(model_name_mat,imageSize,test_folder_1)

close all
clc
clear all
gpu_to_be_used=gpuDevice(1)
model_folder= '../RU-NET/Models/';
addpath(model_folder);
if ~exist('model_name_mat','var'), model_name_mat='res_U_net_DinAbreastlast_R_U-NET500augment0_1_16Copy_of_Inbreast_Train_224.mat'; end     % defualt model folder
if ~exist('imageSize','var'), imageSize=[640 640 3]; end     % defualt test folder
if ~exist('test_folder_1','var'), test_folder_1='INBREAST_MASS_IMAGES/Copy_of_Inbreast_Train_224'; end     % defualt test folder
[f_model,model_name,ext_model]=fileparts(model_name_mat);
mkdir('../RU-NET/Results',model_name);
model_file=fullfile(model_folder,model_name_mat);
network=load (model_file);
% net=network.net;
% net.Layers;
classificationlayer='pixelLabels'; %change if diffrent
% test images:
test_folder_1='INBREAST_MASS_IMAGES/Copy_of_Inbreast_Train_224';
test_folder=strcat('../RU-NET/All_large_patches/',test_folder_1,'/patches');
truth_folder=strcat('../RU-NET/All_large_patches/',test_folder_1,'/labels/');
imds = imageDatastore(test_folder)

addpath(test_folder);
addpath(truth_folder);
all_imgs = dir(fullfile(test_folder,'/*.png'));
all_imgs = {all_imgs.name}';
all_files=all_imgs;
classNames = ["tumour" "background"];
labelIDs   = [255 0];
pxdsTruth = pixelLabelDatastore(truth_folder, classNames, labelIDs);
%Run semantic segmentation on all of the test images.
tempdir=strrep(model_name_mat,'.mat','');
folder='../RU-NET/Test_Modules/';
temp_dir=strcat(folder,tempdir);
mkdir(folder,tempdir);
addpath(temp_dir);
% % %Running semantic segmentation network
% %
% % %Compare results against ground truth.
files = dir(strcat('../RU-NET/All_large_patches/',test_folder_1,'/patches','/*.png'));
averagetime=0;
summ=0;
summ_d2=0;
counter=1;
lengthOfFiles=length(files);
for file = files'
    %../RU-NET/All_large_patches/',test_folder_1,'/patche
    imagepath = strcat('../RU-NET/All_large_patches/',test_folder_1,'/patches/',file.name);
    testImage=imresize(imread(imagepath),imageSize(1:2));
    newStr = strcat('../RU-NET/All_large_patches/',test_folder_1,'/labels/',file.name);
    testLabel =imresize( imread(newStr),imageSize(1:2));
    t1 = cputime;
    tic
    C = semanticseg(imresize(testImage,imageSize(1:2)),net);
    toc
    t2=cputime;
    Timeelapsed = t2-t1
    averagetime=averagetime+Timeelapsed;
    B = labeloverlay(imresize(testImage,imageSize(1:2)),C);
    %
    % figure
    % imshow(B)
    test=testLabel;
    test(C=='background')=0;
    test(C=='tumour')=255;
    % figure;
    %
    % imshow(test)
    % figure;
    testlabel=testLabel;
    testlabel(testLabel==0)=0;
    testlabel(testLabel==255)=255;
    
    
    BinaryUnet=testLabel;
    BinaryUnet(C=='background')=0;
    BinaryUnet(C=='tumour')=1;
    % figure;
    %
    % imshow(test)
    % figure;
    Binarytestlabel=testLabel;
    Binarytestlabel(testLabel==0)=0;
    Binarytestlabel(testLabel==255)=1;
    
    Label_C=C;
    Label_C(testLabel==0)='background';
    
    Label_C(testLabel==255)='tumour';
    file.name
    DC = dice(C,Label_C);
    DCAVARAGE=mean(DC);
    averageaverage=mean(DCAVARAGE)
    DC1 = getDiceCoeff(imresize(BinaryUnet,imageSize(1:2)),imresize(Binarytestlabel,imageSize(1:2)))
    figure;
    subplot(1,3,1), imshow(imresize(testImage,imageSize(1:2))) ,title(file.name);
    %subplot(1,3,2), imshow(imresize(test,imageSize(1:2))) , title(strcat(num2str((DC(1)+DC(2))/2)))% ',num2str(DCAVARAGE)));
    subplot(1,3,2), imshow(imresize(test,imageSize(1:2))) , title(strcat(num2str(DC(1)),'  ',num2str(DC(2))));
    subplot(1,3,3), imshow(imresize(testlabel,imageSize(1:2))),title('Binary image');
    summ=DC+summ;
    summ_d2=DC1+summ_d2;
    display(file.name);
    if(DC>0)
        Y_Test(counter)=1;
        T_Test(counter)=1;
    else
        Y_Test(counter)=0;
        T_Test(counter)=1;
        
    end
    
    counter=counter+1;
end

summ
lengthOfFiles;
averageDice=summ/lengthOfFiles
averageDice2=summ_d2/lengthOfFiles
averagetime=averagetime/lengthOfFiles
%
accuracy = sum(Y_Test == T_Test)/numel(T_Test)
stats = confusionmatStats(T_Test , Y_Test)
[Cc,order] = confusionmat(T_Test , Y_Test)
test_folder=strcat('../RU-NET/All_large_patches/',test_folder_1,'/patches');
truth_folder=strcat('../RU-NET/All_large_patches/',test_folder_1,'/labels/');
imageDirtest = fullfile(test_folder);
labelDirtest = fullfile(truth_folder);
augmentertest = imageDataAugmenter()
imdstest = imageDatastore(imageDirtest);

pxdstest = pixelLabelDatastore(labelDirtest,classNames,labelIDs);
trainingData = pixelLabelImageDatastore(imdstest,pxdstest, ...
    'DataAugmentation',augmentertest,'OutputSize',imageSize,'OutputSizeMode','resize');
tempdir=strrep(model_name_mat,'.mat','');
folder='../RU-NET/Test_Module/';
temp_dir=strcat(folder,tempdir);
mkdir(folder,tempdir);
addpath(temp_dir);
clearvars -global -except imdstest net temp_dir pxdstest


[pxdsResults]= semanticseg(imdstest,net,'WriteLocation',temp_dir);

%Compare results against ground truth.

clearvars  net

metrics = evaluateSemanticSegmentation(pxdsResults,pxdstest)
metrics.DataSetMetrics
metrics.ClassMetrics
metrics.NormalizedConfusionMatrix
normConfMatData = metrics.NormalizedConfusionMatrix.Variables
figure
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';

imageIoU = metrics.ImageMetrics.MeanIoU;
figure
average_mean_iou=mean(imageIoU)
histogram(imageIoU)

title('Mean IOU per image')

disp('end')
reset(gpu_to_be_used)