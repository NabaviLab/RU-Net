function [] = TrainResNet()
% Fine-tune a pretrained convolutional neural network to learn the features
% on a new collection of images.
clc
close all
gpu_to_be_used=gpuDevice(1);
imageSize=[224 224 3];
augment=1
epoches=200;epoches_str=num2str(epoches);
Initial_rate=0.01;learnrate_str=num2str(Initial_rate);
DropFactor=1e-1;
DropPeriod=15;
miniPatchSize=8;
modelName='Binary-scale_ResNet'
test_folder_1='Classification/'; % create folder for output models
digitDatasetPath1 =  '../RU-NET/All_large_patches/Mass_Masks/training_dataset/'; %path to training dat-set
digitData1 = imageDatastore(digitDatasetPath1, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
subDir_Images= dir('./training_dataset/');
digitData1.ReadFcn = @(filename)readAndPreprocessImage(filename,imageSize);
    nn=numel(subDir_Images);
length(digitData1.Files);
CountLabel = digitData1.countEachLabel;
perm = randperm(length(digitData1.Files),20);  
[merchImagesTrain,merchImagesTest,valid] = gpu_to_be_used=gpuDevice(1);
            
 numberOfTraningImages=merchImagesTrain.countEachLabel;
 numberOfTestImages= merchImagesTest.countEachLabel;

if(augment==0)
    imageAugmenter = imageDataAugmenter();
    
    
else
% imageAugmenter = imageDataAugmenter('RandXReflection',true , 'RandYReflection', true ,'RandRotation',[-90 90],...
%                                     'RandXScale',[0.3 4],...
%                                     'RandYScale',[0.3 4],...
%                                     'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
      
    imageAugmenter = imageDataAugmenter('RandXReflection',true , 'RandYReflection', true ,'RandRotation',[-90 90],...
                                    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);                            
end


datasource = augmentedImageSource(imageSize,merchImagesTrain,'DataAugmentation',imageAugmenter)
net =resnet50;

% Determine the number of classes from the training data.
numClasses = numel(categories(merchImagesTrain.Labels));
%%
% Calculate the classification accuracy.
testLabels = merchImagesTest.Labels;

options = trainingOptions('sgdm', ...
    'ValidationData',merchImagesTest, ...
    'ValidationFrequency',30, ...
    'InitialLearnRate',Initial_rate, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',DropFactor,...
    'LearnRateDropPeriod',DropPeriod,...
    'MaxEpochs',epoches,...
    'Shuffle','every-epoch',...
    'MiniBatchSize',miniPatchSize,...
    'Verbose',true,...
    'ExecutionEnvironment','gpu',... 
    'WorkerLoad',36,...
    'Plots','training-progress');

functions = { ...
    @plotTrainingAccuracy, ...?
    @(info) stopTrainingAtThreshold(info,98)
    };
%%
% layersTransfer = net.Layers(1:end-3);
% layers = [...
%     layersTransfer
%     fullyConnectedLayer(numClasses,'Name','fc8')
%     softmaxLayer('Name','prob')
%     classificationLayer('Name','output')]
model_folder_saved='../RU-NET/ModelsClassify/'
MESSAGE="TEST"
modelname=strcat('Classify_Binary_Masks_',modelName,' ','epoches',epoches_str,'augment',num2str(augment),MESSAGE,'.mat')
% Fine-tune the network using |trainNetwork| on the new layer array.
%convnet2 = trainNetwork(XTrain,TTrain,net.Layers,options)
lgraph = layerGraph(net );
FCL=fullyConnectedLayer(2,'Name','fc1000');
last_layer = classificationLayer('Name','ClassificationLayer_fc1000');

lgraph = replaceLayer(lgraph ,'fc1000', FCL);
lgraph = replaceLayer(lgraph ,'ClassificationLayer_fc1000', last_layer);
[netTransfer, info] = trainNetwork(datasource,lgraph,options)
model=fullfile(model_folder_saved,modelname)
save (model,'netTransfer');
meanTraningLoss=mean(info.TrainingLoss,'omitnan')
meanTraningAccuracy=mean(info.TrainingAccuracy,'omitnan')
meanValidationAccuracy=mean(info.ValidationAccuracy,'omitnan')
meanValidationLoss=mean(info.ValidationLoss,'omitnan')
[YPred,scores] = classify(netTransfer,valid);
YTest= valid.Labels;
Testaccuracy = mean(YPred == YTest)
info
model=fullfile(model_folder_saved,modelname)
save (model,'netTransfer');


display("end")


end
function plotTrainingAccuracy(info)

persistent plotObj
persistent Epoch Iteration TrainingLoss BaseLearnRate TrainingAccuracy TrainingRMSE

if info.State == "start"
    plotObj = animatedline;
    xlabel("Iteration")
    ylabel("Training Accuracy")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
else
   % plotTrainingLoss(Iteration,TrainingLoss)
    AverageTrainingAccuracy=mean(TrainingAccuracy(1,:),'omitnan')
end

Epoch=[Epoch info.Epoch];
Iteration=[Iteration info.Iteration];
TrainingLoss=[TrainingLoss info.TrainingLoss];
BaseLearnRate=[BaseLearnRate   info.BaseLearnRate];
TrainingAccuracy=[TrainingAccuracy info.TrainingAccuracy];
TrainingRMSE=[TrainingRMSE info.TrainingRMSE];

% Append accuracy for this iteration

end

function plotTrainingLoss(Iteration,TrainingLoss)

  figure;
    xlabel("Iteration")
    ylabel("Training Loss")
   % plot(Iteration,TrainingLoss)
end
function stop = stopTrainingAtThreshold(info,thr)

stop = false;
if info.State ~= "iteration"
    return
end

persistent iterationAccuracy

% Append accuracy for this iteration
iterationAccuracy = [iterationAccuracy info.TrainingAccuracy];

% Evaluate mean of iteration accuracy and remove oldest entry
if numel(iterationAccuracy) == 50
    stop = mean(iterationAccuracy) > thr;

    iterationAccuracy(1) = [];
end

end
function Iout = readAndPreprocessImage(filename, imageSize)

%       filename : File name
%       imageSize : Resolution of input image
%

I = imread(filename);

if ismatrix(I)
    I = cat(3, I, I, I);
end

Iout = imresize(I, imageSize(1:2));

end