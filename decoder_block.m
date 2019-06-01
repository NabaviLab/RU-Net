function layers=decoder_block(numF,tag)
%Dina Abdelhafiz
%Decoder Block
if (tag=='d0')
layers=[
    batchNormalizationLayer('Name',[tag,'_BN1'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu1'])
    convolution2dLayer(3,numF*2,'Padding','same','Stride',1,'Name',[tag,'_conv1'])
    batchNormalizationLayer('Name',[tag,'_BN2'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu2'])
    convolution2dLayer(3,numF*2,'Padding','same','Stride',1,'Name',[tag,'_conv2'])]; 
elseif (tag=='d1')
layers=[
    batchNormalizationLayer('Name',[tag,'_BN1'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv1'])
    batchNormalizationLayer('Name',[tag,'_BN2'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu2'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv2']) 
    batchNormalizationLayer('Name',[tag,'_upconv_BN'])  
    leakyReluLayer(0.2,'Name',[tag,'_upconv_relu'])
    transposedConv2dLayer(2,numF,'Stride',2,'Name',[tag,'_upconv'])
    depthConcatenationLayer(2,'Name',[tag,'_contac'])];    
else
layers=[
    batchNormalizationLayer('Name',[tag,'_BN1'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv1'])
    batchNormalizationLayer('Name',[tag,'_BN2'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu2'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv2']) 
    batchNormalizationLayer('Name',[tag,'_upconv_BN'])  
    leakyReluLayer(0.2,'Name',[tag,'_upconv_relu'])
    transposedConv2dLayer(2,numF/2,'Stride',2,'Name',[tag,'_upconv'])
    depthConcatenationLayer(2,'Name',[tag,'_contac'])];
end
end