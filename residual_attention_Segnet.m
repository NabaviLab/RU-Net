function [lgraph,networkname] = residual_attention_Segnet(classes,netWidth,height,width,beta)
%Dina Abdelhafiz
networkname = 'residual_attention_Segnet';
lgraph = layerGraph;
%% Encoder
% initial stage
initial_stage = [
    imageInputLayer([height width 3],'Name','input','Normalization','None')
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','convInp0')  
    leakyReluLayer(0.2,'Name','bridge_Inp')
    batchNormalizationLayer('Name','Inp_BN1')
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','convInp')];
lgraph = addLayers(lgraph,initial_stage);
%% stage1
trunk_branch=residual_block_concat(netWidth,'s1',1);
lgraph = addLayers(lgraph,trunk_branch);
attention_skip_connection = residual_attention_5(netWidth,'s1',0);
lgraph = addLayers(lgraph,attention_skip_connection);
% connect pooling and unpooling layers in attention modules
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_4/indices','s1_up_sampling_skip_connection_attention_1/indices');
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_4/size','s1_up_sampling_skip_connection_attention_1/size');
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_3/indices','s1_up_sampling_skip_connection_attention_2/indices');
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_3/size','s1_up_sampling_skip_connection_attention_2/size');
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_2/indices','s1_up_sampling_skip_connection_attention_3/indices');
lgraph = connectLayers(lgraph,'s1_down_sampling_residual_attention_2/size','s1_up_sampling_skip_connection_attention_3/size');
%
lgraph=connectLayers(lgraph,'s1_attention_dotproduct_skip_connection','s1_add/in2');
%% stage 2
trunk_branch=residual_block_concat(netWidth*2,'s2',1);
lgraph = addLayers(lgraph,trunk_branch);
attention_skip_connection = residual_attention_5(netWidth*2,'s2',1);
lgraph = addLayers(lgraph,attention_skip_connection);
% connect pooling and unpooling layers in attention modules
lgraph = connectLayers(lgraph,'s2_down_sampling_residual_attention_3/indices','s2_up_sampling_skip_connection_attention_1/indices');
lgraph = connectLayers(lgraph,'s2_down_sampling_residual_attention_3/size','s2_up_sampling_skip_connection_attention_1/size');
lgraph = connectLayers(lgraph,'s2_down_sampling_residual_attention_2/indices','s2_up_sampling_skip_connection_attention_2/indices');
lgraph = connectLayers(lgraph,'s2_down_sampling_residual_attention_2/size','s2_up_sampling_skip_connection_attention_2/size');
%
lgraph=connectLayers(lgraph,'s2_attention_dotproduct_skip_connection','s2_add/in2');
%% stage 3
trunk_branch=residual_block_concat(netWidth*4,'s3',1);
lgraph = addLayers(lgraph,trunk_branch);
attention_skip_connection = residual_attention_5(netWidth*4,'s3',2);
lgraph = addLayers(lgraph,attention_skip_connection);
% connect pooling and unpooling layers in attention modules
lgraph = connectLayers(lgraph,'s3_down_sampling_residual_attention_2/indices','s3_up_sampling_skip_connection_attention_1/indices');
lgraph = connectLayers(lgraph,'s3_down_sampling_residual_attention_2/size','s3_up_sampling_skip_connection_attention_1/size');
%
lgraph=connectLayers(lgraph,'s3_attention_dotproduct_skip_connection','s3_add/in2');
%% stage 4
trunk_branch=residual_block_concat(netWidth*8,'s4',1);
lgraph = addLayers(lgraph,trunk_branch);
attention_skip_connection = residual_attention_5(netWidth*8,'s4',2);
lgraph = addLayers(lgraph,attention_skip_connection);
% connect pooling and unpooling layers in attention modules
lgraph = connectLayers(lgraph,'s4_down_sampling_residual_attention_2/indices','s4_up_sampling_skip_connection_attention_1/indices');
lgraph = connectLayers(lgraph,'s4_down_sampling_residual_attention_2/size','s4_up_sampling_skip_connection_attention_1/size');
%
lgraph=connectLayers(lgraph,'s4_attention_dotproduct_skip_connection','s4_add/in2');
%% Bridge:
bridge_stage = [
    batchNormalizationLayer('Name','bridge_BN1')  
    leakyReluLayer(0.2,'Name','bridge_relu1')
    convolution2dLayer(3,netWidth*16,'Padding','same','Stride',1,'Name','bridge_conv1')
    batchNormalizationLayer('Name','bridge_BN2')  
    leakyReluLayer(0.2,'Name','bridge_relu2')
    dropoutLayer(0.5,'Name','bridge_dropout1')
    convolution2dLayer(3,netWidth*16,'Padding','same','Stride',1,'Name','bridge_conv2')
    additionLayer(2,'Name','add_bridge')];
lgraph = addLayers(lgraph,bridge_stage);
%
attention_skip_connection = residual_attention_1(netWidth*16,'bridge',3);
lgraph = addLayers(lgraph,attention_skip_connection);
% connect pooling and unpooling layers in attention modules
lgraph = connectLayers(lgraph,'bridge_down_sampling_residual_attention_1/indices','bridge_up_sampling_skip_connection_attention_1/indices');
lgraph = connectLayers(lgraph,'bridge_down_sampling_residual_attention_1/size','bridge_up_sampling_skip_connection_attention_1/size');
%
lgraph=connectLayers(lgraph,'bridge_attention_dotproduct_skip_connection','add_bridge/in2');
%
%% Decoder:
Decoder_stage = [
    decoder_block_shallow(netWidth*16,'d4')
    decoder_block(netWidth*8,'d3')
    decoder_block(netWidth*4,'d2')
    decoder_block(netWidth*2,'d1')
    decoder_block(netWidth*1,'d0')
    batchNormalizationLayer('Name','final_BN')  
    leakyReluLayer(0.2,'Name','final_relu')    
    convolution2dLayer([1 1],classes,'Padding','same','Stride',[1 1],'BiasL2Factor',10,'Name','final_conv');
    softmaxLayer('Name','softmax')
    Fb_loss('fb classification',beta)];
lgraph = addLayers(lgraph,Decoder_stage);
%% connect long skip connections
lgraph=connectLayers(lgraph,'s3_bottleneck','d4_contac/in2');
lgraph=connectLayers(lgraph,'s2_bottleneck','d3_contac/in2');
lgraph=connectLayers(lgraph,'s1_bottleneck','d2_contac/in2');
lgraph=connectLayers(lgraph,'convInp','d1_contac/in2');
%%
lgraph=connectLayers(lgraph,'convInp','s1_BN1');
lgraph=connectLayers(lgraph,'convInp','s1_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s1_bottleneck','s2_BN1');
lgraph=connectLayers(lgraph,'s1_bottleneck','s2_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s2_bottleneck','s3_BN1');
lgraph=connectLayers(lgraph,'s2_bottleneck','s3_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s3_bottleneck','s4_BN1');
lgraph=connectLayers(lgraph,'s3_bottleneck','s4_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s4_bottleneck','bridge_BN1');
lgraph=connectLayers(lgraph,'s4_bottleneck','bridge_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'add_bridge','d4_BN1');
%
lgraph=connectLayers(lgraph,'s1_conv2','s1_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s2_conv2','s2_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s3_conv2','s3_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s4_conv2','s4_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'bridge_conv2','bridge_attention_transition_skip_connection/in2');
%%
%
lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_3/out','s1_add_residual_attention_1/in2');
lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_2/out','s1_add_residual_attention_2/in2');
lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_1/out','s1_add_residual_attention_3/in2');
lgraph=connectLayers(lgraph,'s2_down_sampling_residual_attention_2/out','s2_add_residual_attention_1/in2');
lgraph=connectLayers(lgraph,'s2_down_sampling_residual_attention_1/out','s2_add_residual_attention_2/in2');
lgraph=connectLayers(lgraph,'s3_down_sampling_residual_attention_1/out','s3_add_residual_attention_1/in2');
lgraph=connectLayers(lgraph,'s4_down_sampling_residual_attention_1/out','s4_add_residual_attention_1/in2');
%analyzeNetwork(lgraph);
end