function layers = short_connection(numF,tag,type)
%Dina abdelhafiz
%short skip connections
if (type==0)
layers = [batchNormalizationLayer('Name',[tag,'_BN_skip'])
          convolution2dLayer(3,numF,'Padding','same','Stride',1,'Name',[tag,'_conv_short_skip'])];
    
elseif (type ==1)
layers = [batchNormalizationLayer('Name',[tag,'_BN_skip'])
          convolution2dLayer(3,numF,'Padding','same','Stride',2,'Name',[tag,'_conv_short_skip'])];
end


end