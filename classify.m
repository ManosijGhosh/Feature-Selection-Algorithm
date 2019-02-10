function [per,model]=classify(x,t,x2,t2,chromosome)
    ch = 3;
    if (ch==1)
        [per,model]=nnetwork(x,t,x2,t2,chromosome);
    elseif (ch==2)
        [per,model]=svmClassifier(x,t,x2,t2,chromosome);
    elseif (ch==3)
        [per,model]=knnClassifier(x,t,x2,t2,chromosome);
    else
        [per,model]=nbClassifier(x,t,x2,t2,chromosome);
    end
end