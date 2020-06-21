% visualizeIDM.m
%
% Visualizes an IDM feature vector as images like described in its paper.
% Img_1 : 0 degree
% Img_2 : 45 degree
% Img_3 : 90 degree
% Img_4 : 135 degree
% Img_5 : End points
%
% Kemal Tugrul Yesilbek
%
%
function [  ] = visualizeIDM( f )
    %% Prepare images
    counter = 1;
    for imgNo = 1:5
       for i = 1:12
           for j = 1:12
            image{imgNo}(i,j) = f(counter);
            counter = counter + 1;
           end
       end
    end
    
    sumimg = image{1} + image{2} + image{3} + image{4};
    
    %% Print images
    figure;
    
    subplot(3,3,1);
    imshow(image{3});
    title('0 degrees');
    
    subplot(3,3,2);
    imshow(image{4});
    title('45 degrees');
    
    subplot(3,3,3);
    imshow(image{1});
    title('90 degrees');

    subplot(3,3,4);
    imshow(image{2});
    title('135 degrees');
    
    subplot(3,3,5);
    imshow(image{5});
    title('End points');
    
    subplot(3,3,6);
    imshow(sumimg);
    title('Sum 1:4');
    
    colormap jet
    
    
end























