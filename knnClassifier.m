function [performance,knnModel]=knnClassifier(x,t,x2,t2,chromosome)
    if(sum(chromosome(1,:)==1)~=0)
        x2=x2(:,chromosome(:)==1);
        x=x(:,chromosome(:)==1);
        
        s=size(t,1);
        label=zeros(1,s);
        for i=1:s
            label(1,i)=find(t(i,:),1);
            %{
            if (find(t(i,:),1)==1)
                label(i,1)=1;
            else
                label(i,1)=2;
            end
            %}
        end

        knnModel=fitcknn(x,label,'NumNeighbors',5,'Standardize',1);
        %knnModel=fitcknn(x,label);
        [label,score] = predict(knnModel,x2);
        %label
        
        s=size(t2,1);
        lab=zeros(s,1);
        for i=1:s
            lab(i,1)=find(t2(i,:),1);
            %{
            if (find(t2(i,:),1)==1)
                lab(i,1)=1;
            else
                lab(i,1)=2;
            end
            %}
        end
        %[c,~]=confusion(t2,label);
        %%{
        %size(lab)
        %size(label)
        c = sum(lab ~= label)/s; % mis-classification rate
        %conMat = confusionmat(Y(P.test),C) % the confusion matrix
        %}
        performance=1-c;
        fprintf('Number of features - %d\n',sum(chromosome(1,:)==1));
        fprintf('The correct classification is %f\n',(100*performance));
    else
        performance=0;
    end
end