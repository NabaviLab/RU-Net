function layers=residual_block(numF,tag,type)
%Dina abdelhafiz
% residual_block_same dimensionality
%tag='s1_u1';
%numF=64;
% type: 0 for same dimensionality
% type: 1 for reduceing dimensionality by half
% type: 2 for upsampling dimensionality
if (type==0)
layers = [
    batchNormalizationLayer('Name',[tag,'_BN1'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv1'])
    batchNormalizationLayer('Name',[tag,'_BN2'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu2'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv2'])
    additionLayer(2,'Name',[tag,'_add'])];

elseif (type ==1)
layers = [
    batchNormalizationLayer('Name',[tag,'_BN1'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',2,'Name',[tag,'_conv1'])
    batchNormalizationLayer('Name',[tag,'_BN2'])  
    leakyReluLayer(0.2,'Name',[tag,'_relu2'])
    convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv2'])
    additionLayer(2,'Name',[tag,'_add'])];
   
end


%analyzeNetwork(layers);
end      
  
        

