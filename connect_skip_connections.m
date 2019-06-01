function layers=connect_skip_connections(layers,tag1,tag2)
stage_index=tag2(2);
unit_index=tag2(4);
if (stage_index=='1' && unit_index=='1')
    layers=connectLayers(layers,'convInp','s1u1_skip_BN');
    layers=connectLayers(layers,'s1u1_skip_conv','s1u1_add/in2');
elseif (unit_index=='1')
    first_add=strcat(tag1,'_add');
    second_add=strcat(tag2,'_add/in2'); 
    skip_bn=strcat(tag2,'_skip_BN');
    skip_conv=strcat(tag2,'_skip_conv');
    layers=connectLayers(layers,first_add,skip_bn);
    layers=connectLayers(layers,skip_conv,second_add);        
else
    first_add=strcat(tag1,'_add');
    second_add=strcat(tag2,'_add/in2'); 
    layers=connectLayers(layers,first_add,second_add);        
end
end
