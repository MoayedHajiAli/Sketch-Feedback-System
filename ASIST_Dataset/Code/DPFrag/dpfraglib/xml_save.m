function [] = xml_save(fig,priNum,figNum,PathName,filename)
X = fig.X;
Y = fig.Y;
T = fig.T;

docNode = com.mathworks.xml.XMLUtils.createDocument...
    ('sketch');
docRootNode = docNode.getDocumentElement;
docRootNode.setAttribute('id',char(java.util.UUID.randomUUID));
k = 1;
for i=1:length(X) %stroke
    for j=1:length(X{i})%points on that stroke
        thisElement = docNode.createElement('point');
        uid = java.util.UUID.randomUUID;
        thisElement.setAttribute('id',char(uid));
        thisElement.setAttribute('x',num2str(X{1,i}(j,1)));
        thisElement.setAttribute('y',num2str(Y{1,i}(j,1)));
        thisElement.setAttribute('pressure',char(0));
        thisElement.setAttribute('time',num2str(T{1,i}(j,1)));
        docRootNode.appendChild(thisElement);
        
        mycell{k}.id = uid;
        mycell{k}.x = X{1,i}(j,1);
        mycell{k}.y = Y{1,i}(j,1);
        mycell{k}.t = T{1,i}(j,1);
        k=k+1;
    end
end
last = k-1;

m=1;
for j=1:length(X)
    thisElement = docNode.createElement('stroke');
    thisElement.setAttribute('id',char(java.util.UUID.randomUUID));
    thisElement.setAttribute('visible','true');
    docRootNode.appendChild(thisElement);
       
    for k=1:length(X{i})
        if m<=last
            thisElementPoints = docNode.createElement('arg');
            thisElementPoints.setAttribute('type','point');
            thisElementPoints.appendChild(docNode.createTextNode(sprintf('%c',char(mycell{m}.id))));
            m=m+1;
            thisElement.appendChild(thisElementPoints); 
        end
        
    end    
end

name = strcat('dpfrag_output/',filename,'/primitives_',num2str(priNum),'combinations_fig',num2str(figNum));
name = strcat(PathName,name);
xmlFileName = [name,'.xml'];
xmlwrite(xmlFileName,docNode);
type(xmlFileName);