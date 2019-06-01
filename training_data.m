function [train,validate]=training_data(train_data_folder,validate_data_folder,flag,imageSize)
% data preparation
classNames = ["tumour"; "background"]; %class names
pixelLabelID = [255 ; 0 ];  %class values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ladir_train=strcat('../RU-NET/Copy_2_of_All_large_patches/',train_data_folder,'/labels'); %change the path to traning data folder
imdir_train=strcat('../RU-NET/Copy_2_of_All_large_patches/',train_data_folder,'/patches');
imds_train = imageDatastore(imdir_train);
pxds_train = pixelLabelDatastore(ladir_train,classNames,pixelLabelID);
%imds_train.ReadFcn = @(filename)readAndPreprocessImage(filename,imageSize );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flag for data augmentation, 0:no augmentation, 1= data augmentation:
if(flag==0)
    imageAugmenter = imageDataAugmenter();
    
    
else
imageAugmenter = imageDataAugmenter('RandRotation',[-10 10],...
                                    'RandXScale',[0.3 4],...
                                    'RandYScale',[0.3 4],...
                                    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
                                
end

train = pixelLabelImageDatastore(imds_train,pxds_train,'DataAugmentation',imageAugmenter,'OutputSize',imageSize,'OutputSizeMode','resize');
% validation
ladir_validate=strcat('../RU-NET/Copy_2_of_All_large_patches/',validate_data_folder,'/labels');
imdir_validate=strcat('../RU-NET/Copy_2_of_All_large_patches/',validate_data_folder,'/patches');
imds_validate =imageDatastore(imdir_validate);
%imds_validate.ReadFcn = @(filename)readAndPreprocessImage(filename,imageSize );
pxds_validate = pixelLabelDatastore(ladir_validate,classNames,pixelLabelID);
validate = pixelLabelImageDatastore(imds_validate,pxds_validate,  'OutputSize',imageSize,'OutputSizeMode','resize');
end