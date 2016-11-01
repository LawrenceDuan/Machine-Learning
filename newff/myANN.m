function [] = myANN(k)

    %% Get data
    [x y] = assign1_load();

    %% Randomly seperate data into 10 folds
    [r c] = size(x);
    base = randperm(c);

    % Make 10 folds
    folds = cell(k,1);
    foldSize = floor(c/k);

    for i=1:k-1
        folds{i} = base(1:foldSize);
        base(1:foldSize) = [];
    end
    folds{k} = base;

    %% Training and testing
    BPoutput = cell(k,1);
    for i=1:k
        % Remove test data in inputs and targets
        inputs = x;
        inputs(:, folds{i}) = [];
        targets = y;
        targets(:, folds{i}) = [];

        net = newff(inputs, targets, 10);
        net.trainparam.epochs = 100;
        net.trainParam.lr = 0.1;
        net.trainParam.goal = 0.0000004;

        net = train(net, inputs, targets);
        BPoutput{i} = sim(net, x(:, folds{i}));

        % Replace the original output vector, making the highest value 1 and others 0
		BPoutput{i} = bsxfun(@eq, BPoutput{i}, max(BPoutput{i}, [], 1));
    end
    
    %% Confusion matrix
    % 		    | 1	  2	  3	  4	  5	  6 (Predicted)
	% ===================================
	% (Actual)  |
	% 	1		|					
	% 	2		|
	% 	3		|
	% 	4		|
	% 	5		|
	% 	6		|
	confusion_matrix = zeros(6, 6);

	% Iterate k times, in order to get the x and y of confusion matrix
	for i = 1 : k
		targets = y(:, folds{i});
		for j = 1 : size(BPoutput{i}, 2)
			predicted = find(BPoutput{i}(:, j) == 1);
			actual = find(targets(:, j) == 1);
			confusion_matrix(actual, predicted) = confusion_matrix(actual, predicted) + 1;
		end
	end
	confusion_matrix
end
