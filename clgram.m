function [Q,R] = clgram(A)
% Input: m x n matrix
% Output: the QR decomposition A = QR, where
% Q is an m x n matrix with othogonal columns, and
% R is an n x n upper-triangular matrix.

[numRow, numCols] = size(A);

Q = zeros(numRow, numCols);
R = zeros(numCols, numCols);
R(1,1) = norm(A(:,1));
Q(:,1) = A(:,1) / R(1,1);

for i=2:numCols
    v_i = A(:,i);
    sumproj = 0;
    for j=1:i-1
        R(j,i) = dot(v_i, Q(:,j));
        sumproj = sumproj + R(j,i)*Q(:,j);
    end
    u_i = v_i - sumproj;
    R(i,i) = norm(u_i);
    Q(:,i) = u_i/R(i,i);
end


end

