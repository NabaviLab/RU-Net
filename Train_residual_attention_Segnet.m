function [model] = Train_residual_attention_Segnet(train_data_folder,validate_data_folder,augment,imageSize,epoches,learnrate,MiniBatchSize)
%Dina Abdelhafiz
%Train a Reseduail  U-Net model

clc  %
clear 
close all
gpu_to_be_used=gpuDevice(1); % use if GPU is avaliable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize values
if ~exist('train_data_folder','var'), train_data_folder='trainBCDRINBREAST'; end     % defualt traning folder
if ~exist('validate_data_folder','var'), validate_data_folder='validate'; end    %validation folder
if ~exist('augment','var'), augment=1; end     % augment=0 or 1 
if ~exist('imageSize','var'), imageSize=[640 640 3] ; end    %change the image size
if ~exist('epoches','var'), epoches=120; end     % change traning epoches
if ~exist('learnrate','var'), learnrate=0.1 ; end    %change learn rate
if ~exist('MiniBatchSize','var'), MiniBatchSize=8 ; end    %change mini batch size
if ~exist('optimizer','var'), optimizer='adam' ; end    %change optimizer
if ~exist('encoderDepth','var'), encoderDepth=5 ; end    %change optimizer
DropFactor=1e-1;
DropPeriod=10;
MiniBatchSize=8;
netwidth=96;
beta = 1;
index=3;
numClasses = 2;
index_str=num2str(index);
epoches_str=num2str(epoches);
learnrate_str=num2str(learnrate);
netwidthstr=num2str(netwidth);
betastr=num2str(beta);
augmentstr=num2str(augment);
model_folder_saved='../RU-NET/Models/'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%advise the traning data-set and the validation data-set

[train,validate]=training_data(train_data_folder,validate_data_folder,augment,imageSize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change the traning options
options = training_options(optimizer,Initial_rate,DropFactor,DropPeriod,MaxEpochs,MiniBatchSize,validate);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% weighting the classes
tbl = countEachLabel(train);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
inverseFrequency = 1./frequency
classWeights = median(frequency) ./ frequency;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
last_layer = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',classWeights,'Name','classification');
%last_layer = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',inverseFrequency,'Name','classification');
classWeights = median(frequency) ./ frequency
last_layer = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',classWeights,'Name','classification');
notes=strcat('RA_Segnet','_epoches',epoches_str,'_augment',augmentstr);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train resuidal attention U-Net model
%res_U_net(classes,netWidth,height,width,beta)
[lgraph,networkname]=residual_attention_Segnet(numClasses,netwidth,imageSize(1),imageSize(1),beta);
lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
lgraph.Layers;
%%
%net=load('../RU-NET/Models/res_U_net_DinAbreastlast_last_R_U-NET45augment0_1_16Copy_of_Inbreast_Train_224.mat');
%lgraph=net.net
%lgraph=layerGraph(lgraph)
network_1='network_1_started'
modelname=strcat(networkname,'_',notes,'_',betastr,'_',netwidthstr,'  ',train_data_folder,'.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%start and save model
net= trainNetwork(train,lgraph,options);
model=fullfile(model_folder_saved,modelname);
save(model,'net');
network_1='network_finished';
disp('end')
reset(gpu_to_be_used)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
