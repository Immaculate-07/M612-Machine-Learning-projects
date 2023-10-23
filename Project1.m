%% ---- Project 1 ----
clear 
close all
clc

%% 1. Use the following MATLAB command

x0 = [0:49]/49;
V = fliplr(vander(x0));

%% 2. Create a matrix with the MATLAB command

A = V(:,1:12);

%% 3. Create a vector of values y_i with the MATLAB command

x1 = [0:0.08:3.92];
b = cos(x1)';

%% 
% V - design Matrix or Vandermonde Matrix generated from features
% X - a_n ... a_0
% b - label

%% Question 1
% Dimension of Matrix A
[numRows, numCols] = size(A);
disp('The dimensions of matrix A')
fprintf('The number of Rows = %d\nThe number of Columns = %d\n\n',numRows,numCols)

% Rank of Matrix A
matrix_rank = rank(A);
fprintf('The rank of A = %d\n\n',matrix_rank)

% Condition Number of A
Cond_Num = cond(A);
fprintf('The Condition number of A = %d\n\n',Cond_Num)

%% Question 2
% Degree of interpolating polynomial
% Since there are 'numCols' columns, the degree of the interpolating
%          polynomial is 'numCols-1'
fprintf('The degree of the interpolating polynomial is: %d\n\n',numCols-1)

%% Question 3
% Approach 1
format long

% Normal Equations --- A^T*A*a = A^T*y = y ----
AT_A = A' * A;
AT_b = A' * b;

% Significant digits for Normal equations

m_Normal = log10((0.5)/(cond(AT_A)*eps));
fprintf('The correct digit for Normal equation is:\n')
fprintf('m = %d\n\n',m_Normal)



% ---------Solving for vector 'a(normal equation method)'
a_normal = AT_A \ AT_b;


% Approach 2
% QR decomcosition

[Q,R] = clgram(A);

% Significant digits for QR Decomposition

m_QR = log10((0.5)/(cond(R)*eps));
fprintf('The correct digit for QR decomposition is:\n')
fprintf('m = %d\n\n',m_QR)

y_QR = Q'*b;

% ----------Solving for vector 'a(QR)'
a_QR = R \ y_QR; 

% Approach 3
% QR decomcosition from Matlab

[Q1,R1] = qr(A);

% Significant digits for QR Decomposition

m_QR_m = log10((0.5)/(cond(R1)*eps));
fprintf('The correct digit forQR decomcosition from Matlab is:\n')
fprintf('m = %d\n\n',m_QR_m)

y_QR_m = Q1'*b;

% ----------Solving for vector 'a(QR matlab)'
a_QR_m = R1 \ y_QR_m; 

% Approach 4
% SVD from Matlab
[U,S,V] = svd(A,0);

% ----------Solving for vector 'a(SVD matlab)'
A_dagger = V*inv(S)*U';

% Significant digits for SVD
% Extract the singular values from the diagonal matrix S
singular_values = diag(S);

% Compute the condition number of A
condition_number = max(singular_values) / min(singular_values);

m_SVD = log10((0.5)/(condition_number*eps));
fprintf('The correct digit for SVD from Matlab is:\n')
fprintf('m = %d\n\n',m_SVD)

a_pinv = A_dagger * b;

% Displaying all solution
% disp('Least Square solution using Normal Equation')
% disp(a_normal)
% 
% disp('Least Square solution using QR Classical Gram')
% disp(a_QR)
% 
% disp('Least Square solution using QR Classical Gram Matlab')
% disp(a_QR_m)
% 
% disp('Least Square solution using SVD Pseudo Inv')
% disp(a_pinv)

%% Plotting of Figure 1: Coefficient displayed as points

a_i = 1:1:numCols;
y1 = a_normal;
y2 = a_QR;
y3 = a_QR_m;
y4 = a_pinv;

% Plot the first set of data points
figure(1)
scatter(a_i,y1,'g','filled');
xticks(a_i)
xticklabels({'a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11'})
hold on;
scatter(a_i,y2,'r','filled');
scatter(a_i,y3,'k','filled');
scatter(a_i,y4,'b','filled');

grid on
xlabel('Coefficients of Polynomial, a_{i}')
ylabel('Values of Coefficents a_{i}');
title('Coefficient of the polynomial of each method')
legend('Normal Equation','Classical QR','Matlab QR','Matlab SVD')

hold off;

saveas(gcf,'figure1.png');


%% Plotting Figure 2: Evaluating the interpolating polynomial on the Training set

% Coefficient vector
coefficient_1 = flip(a_normal);
coefficient_2 = flip(a_QR);
coefficient_3 = flip(a_QR_m);
coefficient_4 = flip(a_pinv);

% Range of x - values
x = x1/3.92;

% Evaluate the polynomial at each x-values
y11 = polyval(coefficient_1, x);
y22 = polyval(coefficient_2, x);
y33 = polyval(coefficient_3, x);
y44 = polyval(coefficient_4, x);

% Plot the polynomial
figure(2)
plot(x,y11,'g','LineWidth',1.5);
hold on
plot(x,y22,'r','LineWidth',1.5);
plot(x,y33,'k','LineWidth',1.5);
plot(x,y44,'b','LineWidth',1.5);
plot(x,cos(x1),'m','LineWidth',1.5);

grid on
xlabel('Training data set')
ylabel('Interpolating and original function');
legend({'Normal Equation','Classical QR','Matlab QR','Matlab SVD','Original Function'},'Location','northwest')

hold off;

saveas(gcf,'figure2.png');


%% Evaluating the Residuum

Residuum_normal_2 = norm(y11'-b);
Residuum_normal_inf = norm(y11'-b, inf);
fprintf('For Normal method: the 2-norm is:%d, and the inf norm is:%d\n\n',Residuum_normal_2,Residuum_normal_inf)

Residum_QR_2 = norm(y22'-b);
Residum_QR_inf = norm(y22'-b, inf);
fprintf('For QR method: the 2-norm is:%d, and the inf norm is:%d\n\n',Residum_QR_2,Residum_QR_inf)

Residum_QR_m_2 = norm(y33'-b);
Residum_QR_m_inf = norm(y33'-b, inf);
fprintf('For QR Matlab method: the 2-norm is:%d, and the inf norm is:%d\n\n',Residum_QR_m_2,Residum_QR_m_inf)

Residum_svd_2 = norm(y44'-b);
Residum_svd_inf = norm(y44'-b,inf);
fprintf('For SVD method: the 2-norm is:%d, and the inf norm is:%d\n\n',Residum_svd_2,Residum_svd_inf)


%% Exploring the testing Error

N_values = [10, 50, 100, 200, 500];

errors1 = zeros(size(1,N_values));
errors2 = zeros(size(N_values));
errors3 = zeros(size(N_values));
errors4 = zeros(size(N_values));

%errors = [errors1,errors2,errors3,errors4];


figure(3);
hold on;

% Loop through each value of N
for i = 1:length(N_values)
    N = N_values(i);
    
    % Generate N equally spaces points in [0,1]
    x_bar = linspace(0,3.92,N);
    
    % Compute the corresponding f values using interpolation
    p1 = polyval(coefficient_1, x_bar/3.92);
    p2 = polyval(coefficient_2, x_bar/3.92);
    p3 = polyval(coefficient_3, x_bar/3.92);
    p4 = polyval(coefficient_4, x_bar/3.92);
    
    % Compute the correspoding y values using function cos(x_bar)
    f1 = cos(x_bar/3.92);
    
    %Compute the norm
    error_inf_1 = norm(p1-f1,inf);
    error_inf_2 = norm(p1-f1,inf);
    error_inf_3 = norm(p1-f1,inf);
    error_inf_4 = norm(p1-f1,inf);
    errors1(i) = error_inf_1;
    errors2(i) = error_inf_2;
    errors3(i) = error_inf_3;
    errors4(i) = error_inf_4;
end
        

scatter(N_values,errors1,'g','filled')
scatter(N_values,errors2,'r','filled')
scatter(N_values,errors3,'k','filled')
scatter(N_values,errors4,'b','filled')

grid on
xticks(N_values)
xticklabels({'10','50','100','200','500'})
xlabel('Different N')
ylabel('error ||p-f||_{inf}');
legend({'Normal Equation','Classical QR','Matlab QR','Matlab SVD'},'Location','southeast')

hold off;

saveas(gcf,'figure3.png');



















