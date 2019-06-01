function [lgraph]=add_attention_block(coarsefeatures,finefeatures,destination,attention_tag,lgraph,channels)
attentionlayers = attention_block(attention_tag,channels);
lgraph=addLayers(lgraph,attentionlayers);
lgraph=disconnectLayers(lgraph,coarsefeatures,[destination,'/in2']);
lgraph=connectLayers(lgraph,finefeatures,[attention_tag,'_attention_addition/in1']);
lgraph=connectLayers(lgraph,coarsefeatures,[attention_tag,'_attention_addition/in2']);
lgraph=connectLayers(lgraph,coarsefeatures,[attention_tag,'_attention_transition/in2']);
lgraph=connectLayers(lgraph,[attention_tag,'_attention_dotproduct'],[destination,'/in2']);
end