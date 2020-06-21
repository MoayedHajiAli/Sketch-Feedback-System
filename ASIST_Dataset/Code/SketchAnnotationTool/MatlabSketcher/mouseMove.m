function mouseMove (object, eventdata)
% C = get (gca, 'CurrentPoint');
%
% global mousePosX;
% global mousePosY;
% mousePosX = C(1,1);
% mousePosY = C(1,2);
%
% assignin('base','mousePosX',mousePosX);
% assignin('base','mousePosY',mousePosY);

whichMouseButton = get(gcf,'SelectionType');
global isMouseDown;

if (strcmp(whichMouseButton,'normal') && isMouseDown)
	C = get (gca, 'CurrentPoint');
	mousePos = [C(1,1) C(1,2)];
	
	global annotSketch;
	annotSketch = [annotSketch ; mousePos];
	
	assignin('base','annotSketch',annotSketch);
	
	plot(C(1,1),C(1,2),'r.');
end

end

