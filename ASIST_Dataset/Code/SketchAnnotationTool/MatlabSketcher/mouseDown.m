function  mouseDown( src, evnt )
% whichMouseButton = get(gcf,'SelectionType');
%
% global isSketching;
%
% if strcmp(whichMouseButton,'normal')
%     isSketching = true;
%     assignin('base','isSketching',isSketching);
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
	isMouseDown = true;
	assignin('base','isMouseDown',isMouseDown);
	
end

end

