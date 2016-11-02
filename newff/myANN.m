% K is the number of folds used in cross-validation
function [confusion_matrices, precision_recall_F1, Average_P_R_F1] = assign1_ann(K)

	% Load the data
	[X, Y] = assign1_load();
	[r, c] = size(X);

	% Split the data into K folds
	base = randperm(c);
	folds = cell(K, 1);
	foldSize = floor(c / K);
	for i = 1 : K - 1
		folds{i} = base(1:foldSize);
		base(1:foldSize) = [];
	end	
	folds{K} = base;

	outputs = cell(K, 1);
    confusion_matrices = cell(K,1);
    precision_recall_F1 = cell(K,1);
    
	for i = 1 : K
		% In every iteration, take one fold as validation set, the others as training set
		inputs = X;
		inputs(:, folds{i}) = [];
		targets = Y;
		targets(:, folds{i}) = [];
		
		net = newff(X, Y, [10 10]);
		net.trainParam.epochs = 1000;
		net.trainParam.mc = 0.9;
		net.trainParam.lr = 0.1;
		net.trainParam.goal = 0.0000004;
		net = train(net, inputs, targets);
		outputs{i} = sim(net, X(:, folds{i}));

		% Replace the original output vector, making the highest value 1 and others 0
		outputs{i} = bsxfun(@eq, outputs{i}, max(outputs{i}, [], 1));
        
        % ...
        confusion_matrices{i} = zeros(6, 6);
        targets = Y(:, folds{i});
        for j = 1 : size(outputs{i}, 2)
            predicted = find(outputs{i}(:, j) == 1);
            actual = find(targets(:, j) == 1);
            confusion_matrices{i}(actual, predicted) = confusion_matrices{i}(actual, predicted) + 1;
        end
        
        %% ...
        % ...
        precision_recall_F1{i} = zeros(6,3);
        
        % ...
        for j = 1 : 6
            TP = confusion_matrices{i}(j,j);
            TPFP = sum(confusion_matrices{i}(:,j));
            TPFN = sum(confusion_matrices{i}(j,:));

            % Calculate precision
            Precision = TP/TPFP;
            precision_recall_F1{i}(j,1)=Precision;

            % Calculate recall
            Recall = TP/TPFN;
            precision_recall_F1{i}(j,2)=Recall;
            
            % Calculate F1-measure
            F1 = 2*TP/(TPFP + TPFN);
            precision_recall_F1{i}(j,3)=F1;
        end
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
    Average_P_R_F1 = zeros(6,3);

	% Filling confusion
	for i = 1 : K
		targets = Y(:, folds{i});
		for j = 1 : size(outputs{i}, 2)
			predicted = find(outputs{i}(:, j) == 1);
			actual = find(targets(:, j) == 1);
			confusion_matrix(actual, predicted) = confusion_matrix(actual, predicted) + 1;
		end
	end
    
    for j = 1:6
        TP = confusion_matrix(j,j);
        TPFP = sum(confusion_matrix(:,j));
        TPFN = sum(confusion_matrix(j,:));

        % Calculate precision
        Precision = TP/TPFP;
        Average_P_R_F1(j,1)=Precision;

        % Calculate recall
        Recall = TP/TPFN;
        Average_P_R_F1(j,2)=Recall;
        
        % Calculate F1-measure
        F1 = 2*TP/(TPFP + TPFN);
        Average_P_R_F1(j,3)=F1;
    end
end