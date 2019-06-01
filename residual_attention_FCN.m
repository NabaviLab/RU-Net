function [lgraph,networkname] = residual_attention_FCN(classes,netWidth,height,width,beta)
%Dina Abdelhafiz
networkname = 'residual_attention_FCN';
lgraph = layerGraph;
initial_stage = [
    imageInputLayer([height width 3],'Name','input','Normalization','None')
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','convInp0')  
    leakyReluLayer(0.2,'Name','bridge_Inp')
    batchNormalizationLayer('Name','Inp_BN1')
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','convInp')];
lgraph = addLayers(lgraph,initial_stage);
trunk_branch=residual_block_concat(netWidth,'s1',1);

attention_skip_connection = residual_attention_2(netWidth,'s1',0);
lgraph = addLayers(lgraph,trunk_branch);

lgraph = addLayers(lgraph,attention_skip_connection);

lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_1','s1_add_residual_attention_3/in2');
lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_2','s1_add_residual_attention_2/in2');
lgraph=connectLayers(lgraph,'s1_down_sampling_residual_attention_3','s1_add_residual_attention_1/in2');

skip_main=strcat('s1_add/in2');
skip_attention_node = strcat('s1','_attention_dotproduct_skip_connection');
lgraph=connectLayers(lgraph,skip_attention_node,skip_main);
% stage2
trunk_branch=residual_block_concat(netWidth*2,'s2',1);

attention_skip_connection = residual_attention_2(netWidth*2,'s2',1);
lgraph = addLayers(lgraph,trunk_branch);

lgraph = addLayers(lgraph,attention_skip_connection);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lgraph=connectLayers(lgraph,'s2_down_sampling_residual_attention_1','s2_add_residual_attention_2/in2');
lgraph=connectLayers(lgraph,'s2_down_sampling_residual_attention_2','s2_add_residual_attention_1/in2');
skip_main=strcat('s2','_add/in2');
skip_attention_node = strcat('s2','_attention_dotproduct_skip_connection');
lgraph=connectLayers(lgraph,skip_attention_node,skip_main);

trunk_branch=residual_block_concat(netWidth*4,'s3',1);
attention_skip_connection = residual_attention_2(netWidth*4,'s3',2);
lgraph = addLayers(lgraph,trunk_branch);
lgraph = addLayers(lgraph,attention_skip_connection);

lgraph=connectLayers(lgraph,'s3_down_sampling_residual_attention_1','s3_add_residual_attention_1/in2');
skip_main=strcat('s3','_add/in2');
skip_attention_node = strcat('s3','_attention_dotproduct_skip_connection');
lgraph=connectLayers(lgraph,skip_attention_node,skip_main);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trunk_branch=residual_block_concat(netWidth*8,'s4',1);
attention_skip_connection = residual_attention_2(netWidth*8,'s4',2);
lgraph = addLayers(lgraph,trunk_branch);
lgraph = addLayers(lgraph,attention_skip_connection);
lgraph=connectLayers(lgraph,'s4_down_sampling_residual_attention_1','s4_add_residual_attention_1/in2');
skip_main=strcat('s4','_add/in2');
skip_attention_node = strcat('s4','_attention_dotproduct_skip_connection');
lgraph=connectLayers(lgraph,skip_attention_node,skip_main);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lgraph=connectLayers(lgraph,'convInp','s1_BN1');
lgraph=connectLayers(lgraph,'convInp','s1_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s1_conv2','s1_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s2_conv2','s2_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s3_conv2','s3_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s4_conv2','s4_attention_transition_skip_connection/in2');
lgraph=connectLayers(lgraph,'s1_bottleneck','s2_BN1');
lgraph=connectLayers(lgraph,'s1_bottleneck','s2_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s2_bottleneck','s3_BN1');
lgraph=connectLayers(lgraph,'s2_bottleneck','s3_BN_skip_connection_attention');
lgraph=connectLayers(lgraph,'s3_bottleneck','s4_BN1');
lgraph=connectLayers(lgraph,'s3_bottleneck','s4_BN_skip_connection_attention');
merge_conv0=[
            convolution2dLayer([1 1],netWidth*8,'Padding','same','Stride',[1 1],'Name','bottleneck_x16')
            batchNormalizationLayer('Name','merge_upconv_x16_BN1')  
            leakyReluLayer(0.2,'Name','merge_upconv_x16_relu1')
            transposedConv2dLayer(16,netWidth*8,'Stride',16,'Name','merge_upconv_x16')];

merge_conv1=[
            convolution2dLayer([1 1],netWidth*4,'Padding','same','Stride',[1 1],'Name','bottleneck_x8')
            batchNormalizationLayer('Name','merge_upconv_x8_BN1')  
            leakyReluLayer(0.2,'Name','merge_upconv_x8_relu1')
            transposedConv2dLayer(8,netWidth*4,'Stride',8,'Name','merge_upconv_x8')];

merge_conv2=[
            convolution2dLayer([1 1],netWidth*2,'Padding','same','Stride',[1 1],'Name','bottleneck_x4')
            batchNormalizationLayer('Name','merge_upconv_x4_BN1')  
            leakyReluLayer(0.2,'Name','merge_upconv_x4_relu1')
            transposedConv2dLayer(4,netWidth*2,'Stride',4,'Name','merge_upconv_x4')];

merge_conv3=[
            convolution2dLayer([1 1],netWidth,'Padding','same','Stride',[1 1],'Name','bottleneck_x2')
            batchNormalizationLayer('Name','merge_upconv_x2_BN1')  
            leakyReluLayer(0.2,'Name','merge_upconv_x2_relu1')
            transposedConv2dLayer(2,netWidth,'Stride',2,'Name','merge_upconv_x2')];
merge_conv4=[
            convolution2dLayer([1 1],netWidth,'Padding','same','Stride',[1 1],'Name','bottleneck_x1')
            batchNormalizationLayer('Name','merge_upconv_x1_BN1')  
            leakyReluLayer(0.2,'Name','merge_upconv_x1')];
lgraph = addLayers(lgraph,merge_conv0);
lgraph = addLayers(lgraph,merge_conv1);
lgraph = addLayers(lgraph,merge_conv2);
lgraph = addLayers(lgraph,merge_conv3);
lgraph = addLayers(lgraph,merge_conv4);
lgraph=connectLayers(lgraph,'s4_bottleneck','bottleneck_x16');
lgraph=connectLayers(lgraph,'s3_bottleneck','bottleneck_x8');
lgraph=connectLayers(lgraph,'s2_bottleneck','bottleneck_x4');
lgraph=connectLayers(lgraph,'s1_bottleneck','bottleneck_x2');
lgraph=connectLayers(lgraph,'convInp','bottleneck_x1');
final_merge=[depthConcatenationLayer(5,'Name','merge_contac')
            batchNormalizationLayer('Name','final_BN_1')  
            leakyReluLayer(0.2,'Name','final_relu_1')    
            convolution2dLayer([1 1],netWidth*2,'Padding','same','Stride',[1 1],'Name','bottle_final')
            batchNormalizationLayer('Name','final_BN_2')  
            leakyReluLayer(0.2,'Name','final_relu_2') 
            convolution2dLayer([1 1],classes,'Padding','same','Stride',[1 1],'BiasL2Factor',10,'Name','final_conv');
            softmaxLayer('Name','softmax')
            Fb_loss('fb classification',beta)];
lgraph = addLayers(lgraph,final_merge);
lgraph=connectLayers(lgraph,'merge_upconv_x1','merge_contac/in1');
lgraph=connectLayers(lgraph,'merge_upconv_x8','merge_contac/in2');
lgraph=connectLayers(lgraph,'merge_upconv_x4','merge_contac/in3');
lgraph=connectLayers(lgraph,'merge_upconv_x2','merge_contac/in4');
lgraph=connectLayers(lgraph,'merge_upconv_x16','merge_contac/in5');
end