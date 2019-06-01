classdef dotproductLayer < nnet.layer.Layer
% This is for mixed attention.    
    properties
        % (Optional) Layer properties
        %dims
        %CoarseFeatures
        %Alpha
        %height
        %width
        %submatrices
        % Layer properties go here
    end

    methods
        function layer = dotproductLayer(name)
            % This function must have the same name as the layer
            
            layer.Name = name;
       
            layer.Description = 'This is a custom layer for attention mechanism';
        
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            % 
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            size_x=size(X);
            height=size_x(1);
            width=size_x(2);
            channels=size_x(3);
            half_channels=0.5*channels;
            if (length(size_x)==3)
                X_1 = X(:,:,1:half_channels);
                X_2 = X(:,:,half_channels+1:channels);
                Z=X_1.*X_2;
            else
                minibatch_size=size_x(4);
                X_new=reshape(X,height,width,minibatch_size*channels);
                half_channels_minibatch=0.5*minibatch_size*channels;
                X_1 = X_new(:,:,1:half_channels_minibatch);
                X_2 = X_new(:,:,half_channels_minibatch+1:minibatch_size*channels);
                Z=X_1.*X_2;
                Z=reshape(Z,height,width,half_channels,minibatch_size);
            end

        end

        function [dX] = backward(~,X,~, dZ , ~ )
            % Backward propagate the derivative of the loss function through 
            % the layer
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation
            % Output:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            %         dLdW1, ..., dLdWn - Derivatives of the loss with respect to each
            % 
            % 
            size_x=size(X);
            height=size_x(1);
            width=size_x(2);
            channels=size_x(3);
            half_channels=0.5*channels;
            if (length(size_x)==3)
                X_1 = X(:,:,1:half_channels);
                X_2 = X(:,:,half_channels+1:channels);
                X_new=cat(3,X_2,X_1);
                dZ_new=repmat(dZ,[1 1 2]);
                dX =X_new.*dZ_new;
            else
                minibatch_size=size_x(4);
                X_new=reshape(X,height,width,minibatch_size*channels);
                half_channels_minibatch=0.5*minibatch_size*channels;
                X_1 = X_new(:,:,1:half_channels_minibatch);
                X_2 = X_new(:,:,half_channels_minibatch+1:minibatch_size*channels);
                X_new=cat(3,X_2,X_1);
                X_new=reshape(X_new,height,width,channels,minibatch_size);
                dZ_new=repmat(dZ,[1 1 2]);
                dX =X_new.*dZ_new;
            end

            
            
            % Layer backward function goes here
        end
        
              
    end
end
