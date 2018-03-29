set(0,'defaultTextInterpreter','latex')
%%
sr=50; %sampling rate
t=linspace(0,1,sr);
dt=bsxfun(@minus,t',t);
w=10*2*pi;
beta=40;
co = @(dt) exp(-beta.*dt.^2).*exp(j*dt.*w);
K_block=co(dt);
%%
C=8;
tmp=randn(C,C)+j*randn(C,C);
%S0= tmp'*tmp;
S0 = eye(C);
shat=(linspace(.5,2,C).*exp([0:C-1]/C*2*pi*j));
S1=S0+shat'*shat;
%% generate data
K0=real(kron(S0,K_block));
K1=real(kron(S1,K_block));
sig=.1;
K0n=K0+eye(size(K0))*sig;
K1n=K1+eye(size(K1))*sig;
g0=chol(K0n)';
g1=chol(K1n)';
N=12000;
n0=6000;
x=zeros(numel(t),C,N);
for n=1:N
    if mod(n,1000) == 0
        fprintf('n=%d,\n',n);
    end
    if n<=n0
        x(:,:,n)=reshape(g0*randn(numel(t)*C,1),numel(t),C);
    else
        x(:,:,n)=reshape(g1*randn(numel(t)*C,1),numel(t),C);
    end
end
%%
figure(1);
plot(t,x(:,:,1))
xlabel('Time, \it Seconds')
ylabel('Amplitude')
title('Example Signal')
figure(2);
plot(shat);
xlabel('Real')
ylabel('Imaginary')
title('Best Separating Vector')
%% other figure
figure(3)
mw=co(t-.5);
xt=x(:,:,1);
xi=conv2(xt,mw(:),'same');
pr=xi*shat';
plot(t,real(pr),'k-',...
    t,abs(pr),'b--','LineWidth',3)
set(gca,'FontSize',24)
xlabel('Time, \it Seconds')
ylabel('$h^{(k)}(t)$')
% orient landscape
% print -dpdf circular_invariance




%% 
data = permute(x,[3,2,1]);
label = [zeros(n0,1);ones(N - n0,1)];
shuffle = randperm(N);
data = data(shuffle,:,:);
label = label(shuffle);
N1 = fix(0.7 * N);
N2 = fix(0.8 * N);
train = data(1:N1,:,:);
labtrain = label(1:N1, :,:);
val = data(N1 + 1:N2,:,:);
labval = label(N1+1:N2);
test = data(N2+1:end,:,:);
labtest = label(N2+1:end);
save('./toy.mat','train','val','test','labtrain','labval','labtest');
