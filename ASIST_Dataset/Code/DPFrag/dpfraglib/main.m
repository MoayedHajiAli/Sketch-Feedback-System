% SINCE SUBPARTS OF QUESTION 10 IS SMALL I SUGGEST TO START WORKING WITH
% THAT QUESTION


% WHEN PROGRAM IS RUN, YOU WILL CHOOSE THE XML FROM DATABASE AND PERSON
% FOLDER. FOR THE XML FILE YOU CHOOSE PROGRAM WILL OUTPUT 2,3,4,...
% COMBINATIONS OF SUBPARTS IN THE IMAGE. THE RESULTS WILL BE PLACED UNDER
% DATABASE FOLDER, FOR THE CHOSEN PERSON.
% E.G. DATABASE/SERIKE/XML/DPFRAG_OUTPUT/10 
% THIS IS THE OUTPUT FOR QUESTION 10 WHICH WAS ANSWERED BY SERIKE. 
% primitives_2combinations_fig1.xml means that it is fig1 of the 2 combinations of
% the subparts of the image. fig2 is the 2nd 2 combinations of the subparts
% and so on.

clear;
close all;
figurenum = 1000000; % figure number for the actual drawing each subpart with a different color
[filename,PathName] = uigetfile('Database/*.xml','Select the xml from a persons folder');
[pathstr, name, ext] = fileparts(filename);
xmlfile = strcat(PathName,filename);

strokes = read_sketch(xmlfile);
stroke_num = length(strokes);
indices = get_corners(xmlfile);


p = 1;
c = 1;
for j=1:stroke_num
    cc=hsv(stroke_num*length(indices{1,j})); %assign different colors to parts btx two detected points
    figure(figurenum);
    hold on;
    
    for k=1:length(indices{1,j})-1
        %plot parts btw detected points each with different lines
        %whole constitute the actual drawing
        plot(strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),1), strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),2), 'color', cc(c,:), 'Marker','*');
        X{c} = strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),1);
        Y{c} = strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),2);
        T{c} = strokes(1,j).times(indices{1,j}(1,k):indices{1,j}(1,k+1),1);
        c = c + 1;
    end
    axis equal;
    p = p + 1;
    %plot detected points with black circles
    plot(strokes(1,j).coords(indices{1,j},1), strokes(1,j).coords(indices{1,j},2), 'blacko','LineWidth',3);
end

% this is the part where combinations of 2,3,4.. subparts are found
% first param: primitive number means subparts with 2 primitive will be shown 
% second param: different parts (primitives) in sketch
% X,Y : the coordinates of these primitives

% get k combinations sorted in time for the xml specified above
% for k=1:c-1
%     [imcell] = get_combinations(k,c,X,Y,T);
%     for i=1:length(imcell)
%         %parameters of xmlsave => figure, primi num, fig num, pathname, filename
%         xml_save(imcell{i},k,i,PathName,name);
%     end
% end

%this is to test if written XML is true or not(seems true)

%xmlfile = 'dpfrag_output/primitives_21.xml';
%strokes = read_sketch(xmlfile);
%stroke_num = length(strokes);
%figure(2000);
%for j=1:stroke_num
%    hold on;
%    plot(strokes(1,j).coords(:,1), strokes(1,j).coords(:,2), 'r.');
%    axis equal;
%end