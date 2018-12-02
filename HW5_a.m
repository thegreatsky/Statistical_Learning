load('TrainingSamplesDCT_8_new.mat');
FG = TrainsampleDCT_FG;
BG = TrainsampleDCT_BG;
u_FG = zeros(64,1);
u_BG = zeros(64,1);
sig_cheetah = zeros(64);
sig_grass = zeros(64);
C=8;

%%% cheetah

% covariance initialization
for i=1:250
    u_FG = u_FG+FG(i,:)';
end
u_FG = u_FG/250;
for i=1:250
    temp=FG(i,:)'-u_FG;
    sig_cheetah = sig_cheetah+temp*temp';
end
sig_cheetah=sig_cheetah/250;
sig_ini = diag(diag(sig_cheetah));
sig = zeros(64,64,8);
% for each mixture
for mixture=1:5
    % random initialization
    u_ini = zeros(64,C);
    for i=1:C
        temp = randi(250);
        u_ini(:,i) = FG(temp,:)';
    end
    tem=rand(1,C);
    p_ini=tem/sum(tem);
    % EM process
    p = p_ini;
    u = u_ini;
    for i=1:C
        sig(:,:,i) = sig_ini;
    end
    cnt = 0;
    while cnt<200
        % p
        for idx =1:C
            temp = 0;
            for i=1:250
                temp_de=0;
                for j=1:C
                    temp_de = temp_de + mvnpdf(FG(i,:)', u(:,j), sig(:,:,j)) * p(1,j);
                end
                temp = temp + mvnpdf(FG(i,:)', u(:,idx), sig(:,:,idx)) * p(1, idx) / temp_de;
            end
            sum_hij = temp;
            p(:, idx) = temp/250;
        end
        p = p/sum(p);
        
        % u
        for idx =1:C
            temp = 0;
            for i=1:250
                temp_de=0;
                for j=1:C
                    temp_de = temp_de + mvnpdf(FG(i,:)', u(:,j), sig(:,:,j)) * p(1,j);
                end
                temp = temp + mvnpdf(FG(i,:)', u(:,idx), sig(:,:,idx)) * p(1, idx) * FG(i,:)' / temp_de;
            end
            u(:, idx) = temp/sum_hij;
        end
        
        % sig
        for idx =1:C
            temp = 0;
            for i=1:250
                temp_de=0;
                for j=1:C
                    temp_de = temp_de + mvnpdf(FG(i,:)', u(:,j), sig(:,:,j)) * p(1,j);
                end
                temp = temp + mvnpdf(FG(i,:)', u(:,idx), sig(:,:,idx)) * p(1, idx) * (FG(i,:)' - u(:,idx)) * (FG(i,:)' - u(:,idx))' / temp_de;
            end
            sig(:,:, idx) = temp/sum_hij;
        end
        cnt = cnt+1;
    end
    
    
end
