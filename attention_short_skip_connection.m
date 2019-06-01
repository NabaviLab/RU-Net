function layers = attention_short_skip_connection(numF,tag,type)

if (type==0)
layers = [batchNormalizationLayer('Name',[tag,'_BN_skip_connection_attention'])
          convolution2dLayer(3,numF,'Padding','same','Stride',2,'Name',[tag,'_down_sampling_skip_connection_attention_'])
          transposedConv2dLayer(2,numF,'Stride',2,'Name',[tag,'_up_sampling_skip_connection_attention'])
          leakyReluLayer(0.2,'Name',[tag,'__relu_skip_connection_attention'])
          softmaxLayer('Name',[tag,'_softmax_skip_connection_attention'])
          depthConcatenationLayer(2,'Name',[tag,'_attention_transition_skip_connection'])
          dotproductLayer([tag,'_attention_dotproduct_skip_connection'])
          ];
    
elseif (type ==1)
layers = [batchNormalizationLayer('Name',[tag,'_BN_skip_connection_attention'])
          convolution2dLayer(3,numF,'Padding','same','Stride',2,'Name',[tag,'_down_sampling_skip_connection_attention_'])
          leakyReluLayer(0.2,'Name',[tag,'__relu_skip_connection_attention'])
          softmaxLayer('Name',[tag,'_softmax_skip_connection_attention'])
          depthConcatenationLayer(2,'Name',[tag,'_attention_transition_skip_connection'])
          dotproductLayer([tag,'_attention_dotproduct_skip_connection'])];
end

end