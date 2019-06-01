
clc
clear all
augment=0
train_data_folder='train_data_folder';
validate_data_folder='validate_data_folder'; %case3 adhist + norm new 224
[train,validate]=training_data(train_data_folder,validate_data_folder,augment);
epoches=50;epoches_str=num2str(epoches);
learnrate=0.1;learnrate_str=num2str(learnrate);
options = training_options('adam',learnrate,1e-5,5,epoches,2,validate);
model_folder_saved='../Attentional_FCN-master/Models/';
%ISBI_2018_Tutorial-master-2/Attentional_FCN-master/Models
netwidth=96;
netwidthstr=num2str(netwidth);
% beta=1;beta_str=num2str(beta);
% notes=strcat('_netwidth',netwidthstr,'_train_',train_data_folder,'_validate_',validate_data_folder);
%% weighting the classes
%
tbl = countEachLabel(train)
totalNumberOfPixels = sum(tbl.PixelCount)
frequency = tbl.PixelCount / totalNumberOfPixels;
inverseFrequency = 1./frequency
last_layer = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',inverseFrequency,'Name','classification');
gpu_to_be_used=gpuDevice(3);
%% model 1
beta = 1;betastr=num2str(beta);
%notes=strcat('_trainCase2processed_fb_',betastr,'_');
notes=strcat('_Segnet_50_no_aug_encoderDepth_2_');
index=1;
index_str=num2str(index);
%
networkname='Segnet'
numClasses = 2
encoderDepth=2
imageSize=[448 448 3]
lgraph = segnetLayers(imageSize,numClasses,encoderDepth);
%lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
lgraph = replaceLayer(lgraph,'pixelLabels',last_layer);
%lgraph=fcnLayers([448 448],2,'type','8s');
networkname='Segnet';
%last_layer = Fb_loss('Fb classification',beta);
%lgraph = replaceLayer(lgraph ,'pixelLabels', last_layer);

%%
network_1='network_1_started'
modelname=strcat(networkname,notes,index_str,'.mat')
net= trainNetwork(train,lgraph,options);
model=fullfile(model_folder_saved,modelname);
save (model,'net');
network_1='network_1_finished'
% %% model 2
% beta = 1;betastr=num2str(beta);
% notes=strcat('_trainCase2_width96_');
% index=1;
% index_str=num2str(index);
% [lgraph,networkname]=mixed_attention_U_net(2,netwidth,448,448,beta,4);
% %lgraph=fcnLayers([448 448],2,'type','8s');
% %networkname='FCN8_fb_1_';
% lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
%last_layer = Fb_loss('Fb classification',beta);
%lgraph = replaceLayer(lgraph ,'pixelLabels', last_layer);
% %%
% network_2='network_2_started'
% net= trainNetwork(train,lgraph,options);
% modelname=strcat(networkname,notes,index_str,'.mat')
% model=fullfile(model_folder_saved,modelname);
% save (modelname,'net');
% network_2='network_2_finished'
% %% model 3
% beta = 1;betastr=num2str(beta);
% notes=strcat('_trainCase2_width96_');
% index=1;
% index_str=num2str(index);
% [lgraph,networkname]=residual_attention_Unet_mini_unet(2,netwidth,448,448,beta);
% %lgraph=fcnLayers([448 448],2,'type','8s');
% %networkname='FCN8_fb_1_';
% lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
% %last_layer = Fb_loss('Fb classification',beta);
% %lgraph = replaceLayer(lgraph ,'pixelLabels', last_layer);
% %%
% network_3='network_3_started'
% net= trainNetwork(train,lgraph,options);
% modelname=strcat(networkname,notes,index_str,'.mat')
% model=fullfile(model_folder_saved,modelname);
% save (modelname,'net');
% network_3='network_2_finished'
% %{
% %% model 3
% beta = 1;betastr=num2str(beta);
% notes=strcat('_trainCase2_fb_',betastr,'_');
% index=3;
% index_str=num2str(index);
% [lgraph,networkname]=residual_attention_FCN_conv_4(2,netwidth,448,448,beta);
% %lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
% %%
% net= trainNetwork(train,lgraph,options);
% modelname=strcat(networkname,notes,index_str,'.mat');
% model=fullfile(model_folder_saved,modelname);
% save (modelname,'net');
% %}
% 
% 
% 
% %% model 4
% beta = 6;betastr=num2str(beta);
% %notes=strcat('_trainCase2processed_fb_',betastr,'_');
% notes=strcat('_trainCase2_width96_');
% index=1;
% index_str=num2str(index);
% % [lgraph,networkname]=mixed_attention_U_net(2,netwidth,448,448,beta,3);
% % lgraph = replaceLayer(lgraph ,'fb classification', last_layer);
% 
% lgraph=fcnLayers([448 448],2,'type','8s');
% networkname='FCN8_fb_1_';
% %last_layer = Fb_loss('Fb classification',beta);
% lgraph = replaceLayer(lgraph ,'pixelLabels', last_layer);
% 
% %%
% network_4='network_4_started'
% net= trainNetwork(train,lgraph,options);
% modelname=strcat(networkname,notes,index_str,'.mat')
% model=fullfile(model_folder_saved,modelname);
% save (modelname,'net');
% network_4='network_4_finished'