function mouseUp( src, evnt )
% whichMouseButton = get(gcf,'SelectionType');
% 
% global isSketching;
% global isMouseUp;
% 
% if strcmp(whichMouseButton,'normal')
% 	isSketching = false;
% 	isMouseUp = true;
% 	assignin('base','isSketching',isSketching);
% 	assignin('base','isMouseUp',isMouseUp);
% end

whichMouseButton = get(gcf,'SelectionType');

if strcmp(whichMouseButton,'normal')
	C = get (gca, 'CurrentPoint');
	mousePos = [C(1,1) C(1,2)];

	global annotSketch;
	annotSketch = [annotSketch ; mousePos];

	assignin('base','annotSketch',annotSketch);

	plot(C(1,1),C(1,2),'r.');
	
	global isMouseDown;
	isMouseDown = false;
	assignin('base','isMouseDown',isMouseDown);
	
end


end

