function [inputs, targets] = assign1_load()

		% Load the emotion data and transpose it
		load('emotions_data_66.mat');
		inputs = x';
		targets1 = y';

		% Create a 6*612 matrix to store the labels as vectors
		targets = zeros(6, 612);
		for i = 1 : 612
		   targets(targets1(1,i),i) = 1;
		end
end
