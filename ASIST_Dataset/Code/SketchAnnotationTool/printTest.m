figure;

sk = annotData.sketch;
figure;

for strk = 1: length(sk)
	plot_stroke(sk(strk));
end

title(annotData.label);

waitforbuttonpress;
close all;
