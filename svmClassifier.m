function [performance,svmModel]=svmClassifier(x,t,x2,t2,chromosome)
    if(sum(chromosome(1,:)==1)~=0)
        x2=x2(:,chromosome(:)==1);
        x=x(:,chromosome(:)==1);
        
        s=size(t,1);
        label=zeros(1,s);
        for i=1:s
            label(1,i)=find(t(i,:),1);
            %{
            if (find(t(i,:),1)==1)
                label(1,i)=1;
            else
                label(1,i)=2;
            end
            %}
        end
        
        if max(label)==2
            svmModel=fitcsvm(x,label,'KernelFunction','polynomial','Standardize',true,'ClassNames',[1 2]);
        else
            class=zeros(1,max(label));
            for i=1:max(label)
                class(i)=i;
            end
            temp = templateSVM('Standardize',1,'KernelFunction','linear','Solver','SMO');
            svmModel = fitcecoc(x,label,'Learners',temp,'FitPosterior',1,'ClassNames',class,'Coding','onevsall');
        end
        
        %svmModel=fitcsvm(x,label,'KernelFunction','linear','Standardize',true,'ClassNames',[1 2]);
        [label,~] = predict(svmModel,x2);
        %label
        %svmModel=svmtrain(x,label);
        %label=svmclassify(svmModel,x2);
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