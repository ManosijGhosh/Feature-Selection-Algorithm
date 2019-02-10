function[] =pso()
    tic
    clc
    %%{
    x=load('Data/inputs.mat');   % Training File 
    x=x.inputs;
    t=load('Data/inputs_target.mat');  % Training label
    t=t.inputs_target;
    x2=load('Data/test.mat');   % Training Test File
    x2=x2.test;
    t2=load('Data/test_target.mat');   % Training Test Label
    t2=t2.test_target;
    %}
    %{
    x=importdata('data/Input.xlsx');
    t=importdata('data/target.xlsx');
    %size(t)
    chr=importdata('data/selection.xlsx');
    
    x2=x(chr(:)==1,:);
    t2=t(chr(:)==1,:);
    x=x(chr(:)==0,:);
    t=t(chr(:)==0,:);
    %}
    disp('Imports done');
    
    n=20;   %number of points being considered
    [~,c]=size(x);
    currentPoints=datacreate(n,c);
    rankCurr=zeros(1,n);
    netArrayCurr=cell(1,n);
    
    bestPoints(1:n,1:c)=datacreate(n,c);
    rankBest=zeros(1,n);
    netArrayBest=cell(1,n);
    
    globalBest=zeros(1,c);
    rankGlobal=0;
    
    velocity=zeros(n,c);
    
    iteration=15;
    for i=1:iteration
        fprintf('Iteration - %d\n\n',i);
        [velocity]=updateVelocity(velocity,currentPoints,bestPoints,globalBest);
        [currentPoints,rankCurr,netArrayCurr]=updatePositions(x,t,x2,t2,velocity,currentPoints,rankCurr,netArrayCurr);
        [bestPoints,rankBest,netArrayBest,globalBest,rankGlobal]=updateBest(currentPoints,rankCurr,netArrayCurr,bestPoints,rankBest,globalBest,netArrayBest,rankGlobal);
        
        disp('Current points ---');
        for j=1:n
            fprintf('Feature number - %d  accuracy - %f\n',sum(currentPoints(j,:)==1),rankCurr(1,j));
        end
        
        disp('Best local points ---');
        for j=1:n
            fprintf('Feature number - %d  accuracy - %f\n',sum(bestPoints(j,:)==1),rankBest(1,j));
        end
        
        disp('Best global point ---');
        fprintf('Feature number - %d  accuracy - %f\n',sum(globalBest(:)==1),rankGlobal);
        
        save('result.mat','currentPoints','rankCurr','netArrayCurr','bestPoints','rankBest','netArrayBest');
        disp('results saved');
        disp('----------------------------------------------------------');
    end
    %{
     input=load('Data/Test.mat');
    input=input.testFeature;
    target=load('Data/TestTargets.mat');
    target=target.testTarget;
    
     target=target';
    for i=1:8
        net=netArrayBest{i};
        inputs=input(:,bestPoints(i,:)==1);
        inputs=inputs';
        output=net(inputs);
        %size(output)
        [c, ] = confusion(target,output);
        disp('Final results on test set--- ');
        fprintf('The number of features  : %d\n', sum(bestPoints(i,:)==1));
        fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
        fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
        disp('------------------------------');        
    end
    %}
    toc
end
function [velocity]=updateVelocity(velocity,currentPoints,bestPoints,globalBest)
    rng('shuffle');
    [n,c]=size(velocity);
    for i=1:n
        for j=1:c
            velocity(i,j)=velocity(i,j)+(rand(1)*(bestPoints(i,j)-currentPoints(i,j)))+(rand(1)*(globalBest(1,j)-currentPoints(i,j)));
        end
    end
end
function [currentPoints,rankCurr,netArrayCurr]=updatePositions(x,t,x2,t2,velocity,currentPoints,rankCurr,netArrayCurr)
    rng('shuffle');
    [n,c]=size(velocity);
    for i=1:n
        for j=1:c
            temp=1/(1+exp(-velocity(i,j)));
            if(rand(1)<temp)
                currentPoints(i,j)=1;
            else
                currentPoints(i,j)=0;
            end
        end
        [rankCurr(1,i),netArrayCurr{i}]=classify(x,t,x2,t2,currentPoints(i,:));
    end
end
function [bestPoints,rankBest,netArrayBest,globalBest,rankGlobal]=updateBest(currentPoints,rankCurr,netArrayCurr,bestPoints,rankBest,globalBest,netArrayBest,rankGlobal)
    [n,~]=size(currentPoints);
    for i=1:n
        if(rankCurr(1,i)>rankBest(1,i))
            bestPoints(i,:)=currentPoints(i,:);
            rankBest(1,i)=rankCurr(1,i);
            netArrayBest{i}=netArrayCurr{i};
        end
        if(rankGlobal<rankCurr(1,i))
            globalBest(1,:)=currentPoints(i,:);
            rankGlobal=rankCurr(1,i);
        end
    end
end