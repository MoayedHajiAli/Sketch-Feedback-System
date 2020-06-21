function keyPress( src, evnt )
c = evnt.Character;

global isSketchDone;
isSketchDone = true;
assignin('base','isSketchDone',isSketchDone);

end

