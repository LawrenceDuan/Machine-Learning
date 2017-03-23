function J = ComputeCost(X, y, theta)
    
    m = length(y);
    
    % compute the hypothesis matrix(vector)
    h = X * theta;
    
    % calculate the cost
    J = 1 / (2 * m) * sum((h - y) .^ 2);
end