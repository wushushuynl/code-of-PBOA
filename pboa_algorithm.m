function [AA_pboa, stats_pboa, pboa_curve] = pboa_algorithm()
    clear;
    tic

    % Ԥ�������б������ڴ�ռ�
    numFunctions = 1;
    numRuns = 1;  % ÿ����������50�Σ��޸�Ϊ50�Σ�������һ�£�
    SearchAgents_no = 50;
    Max_iter = 500;

    % Ԥ����������
    AA_pboa = zeros(numFunctions, numRuns);  % �洢ÿ�����е�����ֵ
    stats_pboa = zeros(numFunctions, 4);     % ͳ������ [��ֵ, ��׼��, ��Сֵ, ���ֵ]
    pboa_curve = zeros(numFunctions, Max_iter);  % ��������

    % ���峣������
    p_pop1 = 0.28;
    p_pop2 = 0.56;
    p_pop3 = 0.84;

    for KKK = 1:numFunctions
        % ��ȡ��������
        namestr = ['F' num2str(KKK)];
        [lb, ub, dim, fobj] = Get_Functions_details(namestr);
        
        % ����߽�����
        if length(lb) == 1
            LB = ones(1, dim) * lb;
            UB = ones(1, dim) * ub;
        else
            LB = lb;
            UB = ub;
        end
        
        % ��¼��ǰ������������������
        best_pboa_fit = inf;
        
        for k = 1:numRuns
            % ��ʼ��Alpha, Beta, Gammaλ�úͷ���
            Alpha_pos = zeros(1, dim);
            Alpha_score = inf;
            Beta_pos = zeros(1, dim);
            Beta_score = inf;
            Gama_pos = zeros(1, dim);
            Gama_score = inf;
            
            % ��ʼ����Ⱥλ��
            Positions = LB + (UB - LB) .* rand(SearchAgents_no, dim);
            fitness = zeros(SearchAgents_no, 1);
            converage_curve = zeros(Max_iter, 1);  % ��¼��ǰ���е���������
            
            % ��ʼ����
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
            
            % ����������
            for t = 1:Max_iter
                p = 0.4 * (t/Max_iter);
                
                % �Ż�λ�ø����߼�
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
                    
                    % �߽紦��
                    Positions(i,:) = max(min(Positions(i,:), UB), LB);
                    
                    % ������λ��
                    fitness(i) = fobj(Positions(i,:));
                    
                    % �����쵼��
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
                
                % ��¼��������
                converage_curve(t) = Alpha_score;
            end
            
            % ���浱ǰ���н��
            AA_pboa(KKK, k) = Alpha_score;
            
            % ����������������
            if Alpha_score < best_pboa_fit
                best_pboa_fit = Alpha_score;
                pboa_curve(KKK, :) = converage_curve;
            end
            
            % ��ʾ����
            fprintf('���� %d / %d, ���� %d / %d, PBOA����ֵ: %.6e\n', ...
                    KKK, numFunctions, k, numRuns, Alpha_score);
        end
        
        % ����ͳ��ָ��
        stats_pboa(KKK, :) = [mean(AA_pboa(KKK,:)), std(AA_pboa(KKK,:)), ...
                             min(AA_pboa(KKK,:)), max(AA_pboa(KKK,:))];
    end
    plot(converage_curve)
    toc
end

function [Y] = PL1(X1, X2, ub, lb, t, Max_iter)
    dim = length(X1);
    Y = zeros(1, dim);  % Ԥ�����������
    
    % ��ʼ����һ��ά��
    Y(1) = (X1(1) + X2(1)) / 2;
    
    % �Ƴ������j=1ѭ����ֱ��ִ���ڲ��߼�
    for i = 2 : dim
        % �Ż�C1���㣺�򻯱��ʽ����������С������
        ratio = t / Max_iter;
        C1 = (1 - ratio) * exp(X2(i) - X1(i));  % ԭt-(1e-300)�ɼ�Ϊt�����ɺ���
        C1 = max(C1, 1e-300);  % ��max���if�жϣ�����C1Ϊ0
        
        % ��������������߿ɶ��ԣ�ǰһά���뵱ǰά�ȣ�
        x1_prev = X1(i-1);  % ǰһά�ȵ�X1ֵ
        x2_prev = X2(i-1);  % ǰһά�ȵ�X2ֵ
        y1_curr = X1(i);    % ��ǰά�ȵ�X1ֵ
        y2_curr = X2(i);    % ��ǰά�ȵ�X2ֵ
        
        % ����ֱ�߷��̲�������PL�߼�һ�£�
        A = 2 * (x2_prev - x1_prev);
        B = 2 * (y2_curr - y1_curr);
        C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
        
        % ����A��B��ֵ����Y
        if A == 0
            Y(i-1) = lb(i) + (ub(i) - lb(i)) * rand;  % ǰһά�������ֵ
            Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * (1 - 2 * rand) * C1;
        elseif B == 0
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;  % ��ǰά�������ֵ
            Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * (1 - 2 * rand) * C1;
        else
            Y(i) = ((-A * Y(i-1) - C) / B) * (1 - 2 * rand) * C1;
        end
        
        % �߽������������ϲ�����������ubΪub(i)��
        if Y(i) < lb(i) || Y(i) > ub(i)
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
        end
    end
    
    % �������һ��ά�����һ��ά�ȵ�����
    x1_last = X1(dim);    % ���ά�ȵ�X1
    x2_last = X2(dim);    % ���ά�ȵ�X2
    y1_first = X1(1);     % ��һ��ά�ȵ�X1
    y2_first = X2(1);     % ��һ��ά�ȵ�X2
    
    % �������ӷ��̲���
    A = 2 * (x2_last - x1_last);
    B = 2 * (y2_first - y1_first);
    C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);
    
    % ���µ�һ��ά�ȵ�Yֵ
    if A == 0
        Y(1) = y1_first - (y2_first - y1_first) / 2 * C1 * (1 - 2 * rand);
    elseif B ~= 0  % ����B����ʱ���£����򱣳�ԭֵ
        Y(1) = ((-A * Y(dim-1) - C) / B) * C1 * (1 - 2 * rand);  % i��ʱΪdim��i-1=dim-1
    end
    
    % ������һ��ά�ȵı߽�
    if Y(1) < lb(1) || Y(1) > ub(1)
        Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
    end
end



function [Y] = PL(X1, X2, ub, lb)
    % ��ȡά��
    dim = length(X1);
    
    % Ԥ�����������
    Y = zeros(1, dim);
    
    % ���ó���
    C1 = 1;
    
    % ��ʼ����һ����ΪX1��X2���е�
    Y(1) = (X1(1) + X2(1)) / 2;
    
    % ��ѭ��������ӵڶ��������һ��ά��
    for i = 2:dim
        % ��ȡǰһ��ά�ȵ�����
        x1_prev = X1(i-1);
        x2_prev = X2(i-1);
        
        % ��ȡ��ǰά�ȵ�����
        y1_curr = X1(i);
        y2_curr = X2(i);
        
        % ����ֱ�߷��̲���
        A = 2 * (x2_prev - x1_prev);
        B = 2 * (y2_curr - y1_curr);
        C = (x1_prev^2 - x2_prev^2) + (y1_curr^2 - y2_curr^2);
        
        % ���ݲ������㵱ǰά�ȵ�Yֵ
        if A == 0
            % ����AΪ0�����
            Y(i-1) = lb(i-1) + (ub(i-1) - lb(i-1)) * rand;
            Y(i) = y1_curr - (y2_curr - y1_curr) / 2 * C1;
        elseif B == 0
            % ����BΪ0�����
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
            Y(i-1) = x1_prev - (x2_prev - x1_prev) / 2 * C1;
        else
            % һ�����
            Y(i) = ((-A * Y(i-1) - C) / B) * C1;
        end
        
        % �߽���������
        if Y(i) < lb(i) || Y(i) > ub(i)
            Y(i) = lb(i) + (ub(i) - lb(i)) * rand;
        end
    end
    
    % �������һ�������һ���������
    x1_last = X1(dim);
    x2_last = X2(dim);
    y1_first = X1(1);
    y2_first = X2(1);
    
    % �������ӷ��̲���
    A = 2 * (x2_last - x1_last);
    B = 2 * (y2_first - y1_first);
    C = (x1_last^2 - x2_last^2) + (y1_first^2 - y2_first^2);
    
    % ���ݲ������µ�һ����
    if A == 0
        Y(1) = y1_first - (y2_first - y1_first) / 2 * C1;
    elseif B ~= 0
        % ֻ�е�B��Ϊ0ʱ�Ÿ���Y(1)�����򱣳�ԭֵ
        Y(1) = ((-A * Y(dim) - C) / B) * C1;
    end
    
    % ������һ�����Ƿ��ڱ߽���
    if Y(1) < lb(1) || Y(1) > ub(1)
        Y(1) = lb(1) + (ub(1) - lb(1)) * rand;
    end
end

