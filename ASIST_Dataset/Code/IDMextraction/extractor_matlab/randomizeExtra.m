function [trainSetInput , testSetInput, trainSetOutput , testSetOutput] = randomizeExtra(initIn,initOut,trainPortion)
initialInput=initIn;
initialOutput=initOut;
inputRowSize=size(initialInput,1);
inputColumnSize=size(initialInput,2);
outputColumnSize=size(initialOutput,2);
firstIterationCount=floor(inputRowSize*trainPortion);
restCount=inputRowSize-firstIterationCount;
trainSetInput=zeros(firstIterationCount,inputColumnSize);
trainSetOutput=zeros(firstIterationCount,outputColumnSize);
testSetInput=zeros(restCount,inputColumnSize);
testSetOutput=zeros(restCount,outputColumnSize);

RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
temp=inputRowSize;

for i=1:firstIterationCount,
    r = randi(temp,1,1);
    
    holder=initialInput(r,:);
    initialInput(r,:)=initialInput(temp,:);
    initialInput(temp,:)=holder;
    trainSetInput(i,:)=holder;
    
    holder2=initialOutput(r,:);
    initialOutput(r,:)=initialOutput(temp,:);
    initialOutput(temp,:)=holder2;
    trainSetOutput(i,:)=holder2;
    temp=temp-1;
end

for i=1:restCount,
    r = randi(temp,1,1);
    
    holder=initialInput(r,:);
    initialInput(r,:)=initialInput(temp,:);
    initialInput(temp,:)=holder;
    testSetInput(i,:)=holder;
    
    holder2=initialOutput(r,:);
    initialOutput(r,:)=initialOutput(temp,:);
    initialOutput(temp,:)=holder2;
    testSetOutput(i,:)=holder2;
    temp=temp-1;
end
