%function [performance]=nnetwork(population,id)
function [performance,net]=nnetwork(x,t,x2,t2,chromosome)
    %performance=mod(rand(1),.85);
    
    hiddenLayerSize = 20;  %determins the umber of layers and neurons in hidden layers
    net = patternnet(hiddenLayerSize);
    %%{
    if (sum(chromosome(:)==1)==0)
        performance=0;
        net=null;
    else
        [r,c]=size(t);
        target=t(1:r,1:c);
        %input=x(1:r,chromosome(:)==1);
        [~,sz]=size(chromosome);    %36 is the number of blocks
        if (sz==36)  %for the block selection
            fprintf('Doing block\n');
            sz=sum(chromosome(1:36)==1);
            input=zeros(r,sz*8);    %8- number of features of each block
            j=1;    %j - for where i put in input, k - for where in x,stores the number of chromosome
            for k =1:36
                if chromosome(k)==1
                    for i=1:8
                        input(1:r,j)=x(1:r,((k-1)*8+i));
                        j=j+1;
                    end
                end
            end
        else
            %for normal selection
            %disp('x size');
            %size(x)
            %disp('chromosome check');
           % size(chromosome)
            input=x(1:r,chromosome(:)==1);
        end
        
        inputs = input';
        targets = target';

        %{
        %Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 85/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 0/100;
        %}
        [trainInd,valInd,testInd] = divideind(24000,1:20400,20401:24000);
        net.trainParam.epochs = 700;

        % Train the Network
        [net, ] = train(net,inputs,targets);

        % Test the Network
        [r,c]=size(t2);
        target=t2(1:r,1:c);
        %input=x2(1:r,chromosome(:)==1);
        
        [~,sz]=size(chromosome);
        if (sz==36)  %for the block selection
            %fprintf('Doing block\n');
            sz=sum(chromosome(1:36)==1);
            input=zeros(r,sz*8);
            j=1;    %j - for where i put in input, k - for where in x,stores the number of chromosome
            for k =1:36
                if chromosome(k)==1
                    for i=1:8
                        input(1:r,j)=x2(1:r,((k-1)*8+i));
                        j=j+1;
                    end
                end
            end
        else
            %for normal selection
            input=x2(1:r,chromosome(:)==1);
        end
        
        inputs = input';
        targets = target';
        outputs = net(inputs);
        %outputs
        [c, ] = confusion(targets,outputs);
        fprintf('The number of features  : %d\n', sum(chromosome(:)==1));
        fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
        fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
        performance=1-c;%how much accuracy we get
        % View the Network
        %view(net);
    end
    %}
end