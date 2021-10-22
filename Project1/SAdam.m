function [fall,norg, w] = SAdam(dimension,tol,iter_max,Xtrain,label)
fsz = 20; % fontsize
%% initial guess for parameters
lam = 0.001; % initialization of the regularization parameter
w = ones(dimension,1); % initially all parameters are set to 1
n = size(Xtrain,1);
I = 1:n; % use I to calculate f (all data included)
Ig = I; % use Ig to calculate g, the gradient. (size of Ig is the batch size)
mtemp = zeros(dimension, 1); % initially set temp1 
vtemp = zeros(dimension, 1); % initially set temp2
vbc = zeros(dimension, 1);
mbc = zeros(dimension, 1);
beta1 = 0.9; % set beta_1
beta2 = 0.999; % set beta_2
epsilon = 1e-8 * ones(dimension, 1); % set epsilon
%% random choice of indices
% Define size of batch for random choice of indices
batch_size = 512;
%%
f = qloss(I,Xtrain,label,w,lam);
g = qlossgrad(Ig,Xtrain,label,w,lam);
nor = norm(g);
fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%%
tic % start measuring the CPU time
iter = 2;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
% TODO: DEFINE STEPSIZE 
alpha = 0.05;
% above is learning rate that I chose
while nor > tol && iter < iter_max
    % batch
    Ig = randperm(n, batch_size);
    g = qlossgrad(Ig,Xtrain,label,w,lam);
    % above is for batch
    mtemp = beta1 * mtemp + (1-beta1) * g;
    vtemp = beta2 * vtemp + (1-beta2) * (g .* g);
    mbc = mtemp / (1 - beta1^iter);
    vbc = vtemp / (1 - beta2^iter);
    w = w - alpha * mbc ./ (vbc .^ 1/2 + epsilon);
    f = qloss(I,Xtrain,label,w,lam);
    g = qlossgrad(I,Xtrain,label,w,lam);
    nor = norm(g);     
    fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
    norg(iter) = nor;
    fall(iter) = f;
    iter = iter + 1;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter-1,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
end
%% The objective function
function f = qloss(I,Xtrain,label,w,lam)
f = sum(log(1 + exp(-myquadratic(Xtrain,label,I,w))))/length(I) + 0.5*lam*w'*w;
end
%%
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end
%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end
