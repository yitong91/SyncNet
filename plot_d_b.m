params = load('./result.mat');
params.b = squeeze(params.b);
params.phi = squeeze(params.phi);
figure('Color',[1,1,1]);
%plot(shat,'.-'); hold on;
[C,K] = size(params.b);
b = params.b;
phi = [zeros(K,1);params.phi];
plot(-b.*exp(j*phi),'.-'); hold on;