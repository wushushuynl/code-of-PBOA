clear;
tic

% Preallocate memory for all variables
numFunctions = 27;
numRuns = 50;  % Run each function 50 times (modified to 50 runs to meet requirements)
SearchAgents_no = 50;
Max_iter = 500;

% Preallocate result matrices
AA_pboa = zeros(numFunctions, numRuns);  % Store optimal value of each run
stats_pboa = zeros(numFunctions, 4);     % Statistical data [Mean, Std, Min, Max]
pboa_curve = zeros(numFunctions, Max_iter);  % Convergence curve

% Define constant parameters
p_pop1 = 0.28;
p_pop2 = 0.56;
p_pop3 = 0.84;

for KKK = 1:numFunctions
    % Get function details
    namestr = ['F' num2str(KKK)];
    [lb, ub, dim, fobj] = Get_Functions_details(namestr);
    
    % Handle boundary conditions
    if length(lb) == 1
        LB = ones(1, dim) * lb;
        UB = ones(1, dim) * ub;
    else
        LB = lb;
        UB = ub;
    end
    
    % Record the optimal convergence curve for the current function
    best_pboa_fit = inf;
    
    for k = 1:numRuns
        % Initialize positions and scores of Alpha, Beta, Gamma (leaders)
        Alpha_pos = zeros(1, dim);
        Alpha_score = inf;
        Beta_pos = zeros(1, dim);
        Beta_score = inf;
        Gama_pos = zeros(1, dim);
        Gama_score = inf;
        
        % Initialize population positions
        Positions = LB + (UB - LB) .* rand(SearchAgents_no, dim);
        fitness = zeros(SearchAgents_no, 1);
        converage_curve = zeros(Max_iter, 1);  % Record convergence curve of current run
        
        % Initial evaluation of population fitness
        for i = 1:SearchAgents_no
            fitness(i) = fobj(Positions(i,:));
            
            % Update Alpha, Beta, Gamma based on current fitness
            if fitness(i) < Alpha_score
                Gama_pos = Beta_pos;
                Gama_score = Beta_score;
                Beta_pos = Alpha_pos;
                Beta_score = Alpha_score;
                Alpha_pos = Positions(i,:);
                Alpha_score = fitness(i);
            elseif fitness(i) < Beta_score
                Gama_pos = Beta_pos;
                Gama_score = Beta_score;
                Beta_pos = Positions(i,:);
                Beta_score = fitness(i);
            elseif fitness(i) < Gama_score
                Gama_pos = Positions(i,:);
                Gama_score = fitness(i);
            end
        end
        
        % Main iteration loop
        for t = 1:Max_iter
            p = 0.4 * (t/Max_iter);
            
            % Optimize position update logic
            for i = 1:SearchAgents_no
                p1 = rand;
                
                if rand < p
                    if p1 <= p_pop1
                        Positions(i,:) = PL1(Alpha_pos, Positions(i,:), UB, LB, t, Max_iter);
                    elseif p1 <= p_pop2
                        Positions(i,:) = PL1(Beta_pos, Positions(i,:), UB, LB, t, Max_iter);
                    elseif p1 <= p_pop3
                        Positions(i,:) = PL1(Gama_pos, Positions(i,:), UB, LB, t, Max_iter);
                    else
                        random_idx = randi(SearchAgents_no);
                        Positions(i,:) = PL(Positions(random_idx,:), Positions(i,:), UB, LB);
                    end
                else
                    if p1 <= p_pop1
                        Positions(i,:) = PL(Alpha_pos, Positions(i,:), UB, LB);
                    elseif p1 <= p_pop2
                        Positions(i,:) = PL(Beta_pos, Positions(i,:), UB, LB);
                    elseif p1 <= p_pop3
                        Positions(i,:) = PL(Gama_pos, Positions(i,:), UB, LB);
                    else
                        random_idx = randi(SearchAgents_no);
                        Positions(i,:) = PL(Positions(random_idx,:), Positions(i,:), UB, LB);
                    end
                end
                
                % Boundary handling: ensure positions stay within [LB, UB]
                Positions(i,:) = max(min(Positions(i,:), UB), LB);
                
                % Evaluate fitness of the new position
                fitness(i) = fobj(Positions(i,:));
                
                % Update leaders (Alpha, Beta, Gamma)
                if fitness(i) < Alpha_score
                    Gama_pos = Beta_pos;
                    Gama_score = Beta_score;
                    Beta_pos = Alpha_pos;
                    Beta_score = Alpha_score;
                    Alpha_pos = Positions(i,:);
                    Alpha_score = fitness(i);
                elseif fitness(i) < Beta_score
                    Gama_pos = Beta_pos;
                    Gama_score = Beta_score;
                    Beta_pos = Positions(i,:);
                    Beta_score = fitness(i);
                elseif fitness(i) < Gama_score
                    Gama_pos = Positions(i,:);
                    Gama_score = fitness(i);
                end
            end
            
            % Record convergence curve (optimal score at current iteration)
            converage_curve(t) = Alpha_score;
        end
        
        % Save result of current run
        AA_pboa(KKK, k) = Alpha_score;
        
        % Update the optimal convergence curve for the current function
        if Alpha_score < best_pboa_fit
            best_pboa_fit = Alpha_score;
            pboa_curve(KKK, :) = converage_curve;
        end
        
        % Display progress
        fprintf('Function %d / %d, Run %d / %d, PBOA Optimal Value: %.6e\n', ...
            KKK, numFunctions, k, numRuns, Alpha_score);
    end
    
    % Calculate statistical indicators for the current function
    stats_pboa(KKK, :) = [mean(AA_pboa(KKK,:)), std(AA_pboa(KKK,:)), ...
        min(AA_pboa(KKK,:)), max(AA_pboa(KKK,:))];
end
plot(converage_curve)
toc

function [Y] = PL1(X1, X2, ub, lb, t, Max_iter)
dim = length(X1);
Y = zeros(1, dim);  % Preallocate output vector

% Initialize the first dimension
Y(1) = (X1(1) + X2(1)) / 2;

% Remove redundant j=1 loop, execute core logic directly
for i = 2 : dim
    % Optimize C1 calculation: simplify expression, avoid redundant small-value handling
    ratio = t / Max_iter;
    C1 = (1 - ratio) * exp(X2(i) - X1(i));  % Original t-(1e-300) simplified to t (negligible error)
    C1 = max(C1, 1e-300);  % Use max() instead of if-statement to prevent C1 from being 0
    
    % Rename variables for readability (previous dimension vs current dimension)
    x1_prev = X1(i-1);  % X1 value of the previous dimension
    x2_prev = X2(i-1);  % X2 value of the previous dimension
    y1_curr = X1(i);    % X1 value of the current dimension
    y2_curr = X2(i);    % X2 value of the current dimension
    
    % Calculate parameters of the linear equation (consistent with PL logic)
    A = 2 * (x2_prev - x1_prev);
    B = 2 * (y2_curr - y1_curr);
    C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
    
    % Update Y based on values of A and B
    if A == 0
        Y(i-1) = lb(i) + (ub(i) - lb(i)) * rand;  % Random value for the previous dimension
        Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * (1 - 2 * rand) * C1;
    elseif B == 0
        Y(i) = lb(i) + (ub(i) - lb(i)) * rand;  % Random value for the current dimension
        Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * (1 - 2 * rand) * C1;
    else
        Y(i) = ((-A * Y(i-1) - C) / B) * (1 - 2 * rand) * C1;
    end
    
    % Boundary check and correction (combine conditions, fix ub to ub(i))
    if Y(i) < lb(i) || Y(i) > ub(i)
        Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
    end
end

% Handle connection between the last dimension and the first dimension
x1_last = X1(dim);    % X1 of the last dimension
x2_last = X2(dim);    % X2 of the last dimension
y1_first = X1(1);     % X1 of the first dimension
y2_first = X2(1);     % X2 of the first dimension

% Calculate parameters of the connection equation
A = 2 * (x2_last - x1_last);
B = 2 * (y2_first - y1_first);
C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);

% Update Y value of the first dimension
if A == 0
    Y(1) = y1_first - (y2_first - y1_first) / 2 * C1 * (1 - 2 * rand);
elseif B ~= 0  % Update only when B is non-zero; keep original value otherwise
    Y(1) = ((-A * Y(dim-1) - C) / B) * C1 * (1 - 2 * rand);  % i = dim at this point, so i-1 = dim-1
end

% Final boundary check for the first dimension
if Y(1) < lb(1) || Y(1) > ub(1)
    Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
end
end



function [Y] = PL(X1, X2, ub, lb)
% Get the number of dimensions
dim = length(X1);

% Preallocate output vector
Y = zeros(1, dim);

% Set constant parameter
C1 = 1;

% Initialize the first point as the midpoint of X1 and X2
Y(1) = (X1(1) + X2(1)) / 2;

% Main loop: process dimensions from the second to the last
for i = 2:dim
    % Get coordinates of the previous dimension
    x1_prev = X1(i-1);
    x2_prev = X2(i-1);
    
    % Get coordinates of the current dimension
    y1_curr = X1(i);
    y2_curr = X2(i);
    
    % Calculate parameters of the linear equation
    A = 2 * (x2_prev - x1_prev);
    B = 2 * (y2_curr - y1_curr);
    C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
    
    % Calculate Y value of current dimension based on parameters
    if A == 0
        % Handle case when A = 0
        Y(i-1) = lb(i-1) + (ub(i-1) - lb(i-1)) * rand;
        Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * C1;
    elseif B == 0
        % Handle case when B = 0
        Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
        Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * C1;
    else
        % General case
        Y(i) = ((-A * Y(i-1) - C) / B) * C1;
    end
    
    % Boundary check and correction
    if Y(i) < lb(i) || Y(i) > ub(i)
        Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
    end
end

% Handle connection between the last point and the first point
x1_last = X1(dim);
x2_last = X2(dim);
y1_first = X1(1);
y2_first = X2(1);

% Calculate parameters of the connection equation
A = 2 * (x2_last - x1_last);
B = 2 * (y2_first - y1_first);
C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);

% Update the first point based on parameters
if A == 0
    Y(1) = y1_first - (y2_first - y1_first) / 2 * C1;
elseif B ~= 0
    % Update Y(1) only when B is non-zero; keep original value otherwise
    Y(1) = ((-A * Y(dim) - C) / B) * C1;
end

% Final check if the first point is within boundaries
if Y(1) < lb(1) || Y(1) > ub(1)
    Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
end
end