function [fall,norg, w] = SGD(dimension,tol,iter_max,Xtrain,label)
fsz = 20; % fontsize
%% initial guess for parameters
lam = 0.001; % initialization of the regularization parameter
w = ones(dimension,1); % initially all parameters are set to 1
n = size(Xtrain,1);
I = 1:n; % use I to calculate f (all data included)
Ig = I; % use Ig to calculate g, the gradient. (size of Ig is the batch size)
%%
f = qloss(I,Xtrain,label,w,lam);
g = qlossgrad(Ig,Xtrain,label,w,lam);
nor = norm(g);
fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%% The SG method
tic % start measuring the CPU time
iter = 0;
k = 1;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
% TODO: DEFINE STEPSIZE 
% alpha = 1;
% above is learning rate that I chose
% Define size of batch for random choice of indices
batch_size = 512;
batch_ind = randperm(n, batch_size);
alpha_0 = 0.3;
% above is what I set for batch size
while k < iter_max && nor > tol
    % TODO: insert the stochastic gradient descend algorithm here 
    % below is what I add
%     kstep = ceil (2 ^ k / k);
%     alpha_k = 2 ^ (-k) * alpha;
        alpha_k = k ^ -1 * alpha_0;
%     for ii = 1:kstep
        w = w - alpha_k * g;
        f = qloss(I,Xtrain,label,w,lam);
        Ig = batch_ind;
        g = qlossgrad(Ig,Xtrain,label,w,lam);
        nor = norm(g);
    % above is what I add     
        fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
        iter = iter + 1;
        norg(iter+1) = nor;
        fall(iter+1) = f;
%     end
    k = k + 1;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
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
