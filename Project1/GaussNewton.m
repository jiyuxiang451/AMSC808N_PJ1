function [fall,norg,w] = GaussNewton(dimension,tol,iter_max,Xtrain,label)
fsz = 20; % fontsize
%%
eta = 0.01;
gam = 0.9;
% iter_max = 100;
% tol = 5e-3;
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
%% initial guess for parameters
w = ones(dimension,1);
%%
[r,J] = Res_and_Jac(Xtrain, label,w);
f = F(r);
g = J'*r;
nor = norm(g);
fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%% The Gauss--Newton method
tic

iter = 0;
I = eye(length(w));
% quadratic model: m(p) = (1/2)||r||^2 + p'*J'*r + (1/2)*p'*J'*J*p;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
norg(1) = nor;
fall(1) = f;
while nor > tol && iter < iter_max
    B = J'*J + (1e-6)*I;
    p = -B\g;
    a = 1;
    aux = eta*g'*p;
    for j = 0 : jmax
        wtry = w + a*p;
        [rtry, Jtry] = Res_and_Jac(Xtrain,label,wtry);
        f1 = F(rtry);
        if f1 < f + a*aux
            break;
        else
            a = a*gam;
        end
    end
    w = w + a*p;
    [r,J] = Res_and_Jac(Xtrain, label,w);
    f = F(r);
    g = J'*r;
    nor = norm(g);
    fprintf('iter %d: line search: j = %d, a = %d, f = %d, norg = %d\n',iter,j,a,f,nor);
    iter = iter + 1;
    norg(iter+1) = nor;
    fall(iter+1) = f;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
end
%%
function f = F(r)
    f = 0.5*r'*r;
end