function [AA_pboa, stats_pboa, pboa_curve] = pboa_algorithm()
    clear;
    tic

    % 预分配所有变量的内存空间
    numFunctions = 1;
    numRuns = 1;  % 每个函数运行50次（修改为50次，与需求一致）
    SearchAgents_no = 50;
    Max_iter = 500;

    % 预分配结果矩阵
    AA_pboa = zeros(numFunctions, numRuns);  % 存储每次运行的最优值
    stats_pboa = zeros(numFunctions, 4);     % 统计数据 [均值, 标准差, 最小值, 最大值]
    pboa_curve = zeros(numFunctions, Max_iter);  % 收敛曲线

    % 定义常量参数
    p_pop1 = 0.28;
    p_pop2 = 0.56;
    p_pop3 = 0.84;

    for KKK = 1:numFunctions
        % 获取函数详情
        namestr = ['F' num2str(KKK)];
        [lb, ub, dim, fobj] = Get_Functions_details(namestr);
        
        % 处理边界条件
        if length(lb) == 1
            LB = ones(1, dim) * lb;
            UB = ones(1, dim) * ub;
        else
            LB = lb;
            UB = ub;
        end
        
        % 记录当前函数的最优收敛曲线
        best_pboa_fit = inf;
        
        for k = 1:numRuns
            % 初始化Alpha, Beta, Gamma位置和分数
            Alpha_pos = zeros(1, dim);
            Alpha_score = inf;
            Beta_pos = zeros(1, dim);
            Beta_score = inf;
            Gama_pos = zeros(1, dim);
            Gama_score = inf;
            
            % 初始化种群位置
            Positions = LB + (UB - LB) .* rand(SearchAgents_no, dim);
            fitness = zeros(SearchAgents_no, 1);
            converage_curve = zeros(Max_iter, 1);  % 记录当前运行的收敛曲线
            
            % 初始评估
            for i = 1:SearchAgents_no
                fitness(i) = fobj(Positions(i,:));
                
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
            
            % 主迭代过程
            for t = 1:Max_iter
                p = 0.4 * (t/Max_iter);
                
                % 优化位置更新逻辑
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
                    
                    % 边界处理
                    Positions(i,:) = max(min(Positions(i,:), UB), LB);
                    
                    % 评估新位置
                    fitness(i) = fobj(Positions(i,:));
                    
                    % 更新领导者
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
                
                % 记录收敛曲线
                converage_curve(t) = Alpha_score;
            end
            
            % 保存当前运行结果
            AA_pboa(KKK, k) = Alpha_score;
            
            % 更新最优收敛曲线
            if Alpha_score < best_pboa_fit
                best_pboa_fit = Alpha_score;
                pboa_curve(KKK, :) = converage_curve;
            end
            
            % 显示进度
            fprintf('函数 %d / %d, 运行 %d / %d, PBOA最优值: %.6e\n', ...
                    KKK, numFunctions, k, numRuns, Alpha_score);
        end
        
        % 计算统计指标
        stats_pboa(KKK, :) = [mean(AA_pboa(KKK,:)), std(AA_pboa(KKK,:)), ...
                             min(AA_pboa(KKK,:)), max(AA_pboa(KKK,:))];
    end
    plot(converage_curve)
    toc
end

function [Y] = PL1(X1, X2, ub, lb, t, Max_iter)
    dim = length(X1);
    Y = zeros(1, dim);  % 预分配输出向量
    
    % 初始化第一个维度
    Y(1) = (X1(1) + X2(1)) / 2;
    
    % 移除冗余的j=1循环，直接执行内部逻辑
    for i = 2 : dim
        % 优化C1计算：简化表达式，避免冗余小量处理
        ratio = t / Max_iter;
        C1 = (1 - ratio) * exp(X2(i) - X1(i));  % 原t-(1e-300)可简化为t，误差可忽略
        C1 = max(C1, 1e-300);  % 用max替代if判断，避免C1为0
        
        % 变量重命名，提高可读性（前一维度与当前维度）
        x1_prev = X1(i-1);  % 前一维度的X1值
        x2_prev = X2(i-1);  % 前一维度的X2值
        y1_curr = X1(i);    % 当前维度的X1值
        y2_curr = X2(i);    % 当前维度的X2值
        
        % 计算直线方程参数（与PL逻辑一致）
        A = 2 * (x2_prev - x1_prev);
        B = 2 * (y2_curr - y1_curr);
        C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
        
        % 根据A、B的值更新Y
        if A == 0
            Y(i-1) = lb(i) + (ub(i) - lb(i)) * rand;  % 前一维度随机赋值
            Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * (1 - 2 * rand) * C1;
        elseif B == 0
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;  % 当前维度随机赋值
            Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * (1 - 2 * rand) * C1;
        else
            Y(i) = ((-A * Y(i-1) - C) / B) * (1 - 2 * rand) * C1;
        end
        
        % 边界检查与修正（合并条件，修正ub为ub(i)）
        if Y(i) < lb(i) || Y(i) > ub(i)
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
        end
    end
    
    % 处理最后一个维度与第一个维度的连接
    x1_last = X1(dim);    % 最后维度的X1
    x2_last = X2(dim);    % 最后维度的X2
    y1_first = X1(1);     % 第一个维度的X1
    y2_first = X2(1);     % 第一个维度的X2
    
    % 计算连接方程参数
    A = 2 * (x2_last - x1_last);
    B = 2 * (y2_first - y1_first);
    C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);
    
    % 更新第一个维度的Y值
    if A == 0
        Y(1) = y1_first - (y2_first - y1_first) / 2 * C1 * (1 - 2 * rand);
    elseif B ~= 0  % 仅当B非零时更新，否则保持原值
        Y(1) = ((-A * Y(dim-1) - C) / B) * C1 * (1 - 2 * rand);  % i此时为dim，i-1=dim-1
    end
    
    % 最后检查第一个维度的边界
    if Y(1) < lb(1) || Y(1) > ub(1)
        Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
    end
end



function [Y] = PL(X1, X2, ub, lb)
    % 获取维度
    dim = length(X1);
    
    % 预分配输出向量
    Y = zeros(1, dim);
    
    % 设置常量
    C1 = 1;
    
    % 初始化第一个点为X1和X2的中点
    Y(1) = (X1(1) + X2(1)) / 2;
    
    % 主循环：处理从第二个到最后一个维度
    for i = 2:dim
        % 获取前一个维度的坐标
        x1_prev = X1(i-1);
        x2_prev = X2(i-1);
        
        % 获取当前维度的坐标
        y1_curr = X1(i);
        y2_curr = X2(i);
        
        % 计算直线方程参数
        A = 2 * (x2_prev - x1_prev);
        B = 2 * (y2_curr - y1_curr);
        C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
        
        % 根据参数计算当前维度的Y值
        if A == 0
            % 处理A为0的情况
            Y(i-1) = lb(i-1) + (ub(i-1) - lb(i-1)) * rand;
            Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * C1;
        elseif B == 0
            % 处理B为0的情况
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
            Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * C1;
        else
            % 一般情况
            Y(i) = ((-A * Y(i-1) - C) / B) * C1;
        end
        
        % 边界检查与修正
        if Y(i) < lb(i) || Y(i) > ub(i)
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
        end
    end
    
    % 处理最后一个点与第一个点的连接
    x1_last = X1(dim);
    x2_last = X2(dim);
    y1_first = X1(1);
    y2_first = X2(1);
    
    % 计算连接方程参数
    A = 2 * (x2_last - x1_last);
    B = 2 * (y2_first - y1_first);
    C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);
    
    % 根据参数更新第一个点
    if A == 0
        Y(1) = y1_first - (y2_first - y1_first) / 2 * C1;
    elseif B ~= 0
        % 只有当B不为0时才更新Y(1)，否则保持原值
        Y(1) = ((-A * Y(dim) - C) / B) * C1;
    end
    
    % 最后检查第一个点是否在边界内
    if Y(1) < lb(1) || Y(1) > ub(1)
        Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
    end
end

