classdef Fb_loss < nnet.layer.ClassificationLayer
    properties
        betavalue
        

        % Layer properties go here
    end
% laplace smoothed version of fb loss
    methods
        function layer = Fb_loss(name,beta)           
            % (Optional) Create a myClassificationLayer
            
            % Set layer name
            %if nargin == 1
            layer.Name = name;
            %end
            
            layer.betavalue=beta;
            % Set layer description
            layer.Description = 'F beta score cost function layer for semantic segmentation';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here
            obversations=size(Y,4)*size(Y,1)*size(Y,2);
            b=layer.betavalue;                        
            p0=Y;
            p1=-1*Y+1;
            g0=T.^2;
            g1=1-g0;
            
            p0_g0=p0.*g0;
            p0_g1=p0.*g1;
            p1_g0=p1.*g0;
            
            sum_p0_g0=sum(sum(sum(sum(p0_g0,3),2),1))/obversations;
            sum_p0_g1=sum(sum(sum(sum(p0_g1,3),2),1))/obversations;
            sum_p1_g0=sum(sum(sum(sum(p1_g0,3),2),1))/obversations;
            
            fb_numerator=sum_p0_g0*(1+b^2);
            fb_denominator=sum_p0_g0*(1+b^2)+sum_p1_g0*(b^2)+sum_p0_g1;
            fb_coe=(fb_numerator+1)/(fb_denominator+1);
            loss=1-fb_coe;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y

            % Layer backward loss function goes here
            obversations=size(Y,4)*size(Y,1)*size(Y,2);
            b=layer.betavalue;                        
            p0=Y;
            p1=-1*Y+1;
            g0=T.^2;
            g1=1-g0;
            
            p0_g0=p0.*g0;
            p0_g1=p0.*g1;
            p1_g0=p1.*g0;
            
            sum_p0_g0=sum(sum(sum(sum(p0_g0,3),2),1))/obversations;
            sum_p0_g1=sum(sum(sum(sum(p0_g1,3),2),1))/obversations;
            sum_p1_g0=sum(sum(sum(sum(p1_g0,3),2),1))/obversations;
             
            gradient_numerator=((1+b^2)*((b^2)*sum_p1_g0+sum_p0_g1+1)-1)*T;
            gradient_denominator=(sum_p0_g0*(1+b^2)+sum_p1_g0*(b^2)+sum_p0_g1+1).^2;
            dLdY=-(gradient_numerator/gradient_denominator);
            
        end
    end

end