function layers = convolutionalUnit(numF,stride,tag,dropout_ratio,type)
switch type
    case 'ConNormRelu'
        layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    dropoutLayer(dropout_ratio,'Name',[tag,'dropout_medium'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
    case 'ReluNormCon'
         layers = [
    reluLayer('Name',[tag,'relu1'])
    batchNormalizationLayer('Name',[tag,'BN1'])    
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    dropoutLayer(dropout_ratio,'Name',[tag,'dropout_medium1'])
    reluLayer('Name',[tag,'relu2'])
    batchNormalizationLayer('Name',[tag,'BN2'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])];
    case 'NormReluCon'
         layers = [
    batchNormalizationLayer('Name',[tag,'BN1'])  
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    %dropoutLayer(dropout_ratio,'Name',[tag,'dropout_medium1'])
    batchNormalizationLayer('Name',[tag,'BN2'])
    reluLayer('Name',[tag,'relu2'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])];        

end
        

end