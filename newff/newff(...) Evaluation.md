# newff(...) Evaluation

Now that you have a basic understanding of Neural Networks train a network using the emotion dataset from the previous assignment. You can use either the nntool or the command line functions as described below. Next, evaluate the neural networks using 10-fold cross validation. To do so, we will no longer use the graphical nntool. Instead, use the function newff to create a novel network. Read the help pages of newff, network/train and network/sim. Note that you can set the NET.trainParam.show, NET.trainParam.epochs and NET.trainParam.goal values of your network prior to training. The values of these parameters will influence classification performance and training time.

• NET = newff(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF) takes 
• P - RxQ1 matrix of Q1 representative R-element input vectors. 
• T - SNxQ2 matrix of Q2 representative SN-element target vectors. 
• Si - Sizes of N-1 hidden layers, S1 to S(N-1), default is []. 
• TFi - Transfer function of ith layer. Default is 'tansig' for hidden layers, and 'purelin' for output layer. 
• BTF - Backprop network training function, default is 'trainlm'. 
• BLF - Backprop weight/bias learning function, default is 'learngdm'. 
• PF - Performance function, default is 'mse'. 
• IPF - Row cell array of input processing functions. Default is {'fixunknowns','remconstantrows','mapminmax'}. 
• OPF - Row cell array of output processing functions. Default is {'remconstantrows','mapminmax'}. 
• DDF - Data division function, default is 'dividerand'; and returns a N layer feed-forward backprop network.

• [NET, TR] = train(NET, X, T) takes a network NET, input features X and targets T and returns the network after training it, and a training record TR.

• t = sim(NET, X) takes a network NET and input features X and returns the predicted labels t generated by the network.

For more information regarding the above methods please refer to the Matlab help. Make a 10-fold cross-validation evaluation of the neural networks using the parameters and learning rule that is optimal according to you.

(Hint: 10-fold cross validation using neural networks will be used again in the last assignment so you may wish to write a function that takes as inputs the predicted labels t and the targets and returns a 6x6 confusion matrix).

