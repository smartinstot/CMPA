%%%%% ELEC 4700 - Diode Paramater Extraction PA %%%%%
%%%%%%%%%%%%%%%%% March 20th,  2019 %%%%%%%%%%%%%%%%%

rng(234);

% Part 1 - Generate Rangdom Test Data
Is = 0.01E-12;  % A
Ib = 0.1E-12;   % A
Vb = 1.3;       % V
Gp = 0.1;       % Mho

V = linspace(-1.95, 0.7, 20000)';
I_actual = Is*(exp((1.2/0.025)*V)) + Gp*V + Ib*exp(-(1.2/0.25)*(V+Vb));
I_test = I_actual.*(1+0.2*rand(length(I_actual),1));

f_1 = figure('Name', 'Part 1');
hold on;
plot(V,I_actual);
plot(V,I_test);
title('Test Data');
legend('Actual', 'Noise');
xlabel('Voltage (V)');
ylabel('Current (A)');




% Part 2 - Ploynomial Fit
p4 = polyfit(V, I_test, 4);
p8 = polyfit(V, I_test, 8);

I_p4 = polyval(p4,V);
I_p8 = polyval(p8,V);

f_2 = figure('Name', 'Part 2');
hold on;

subplot(1,2,1);
title('4 Term Polynomial Fit');
hold on;
xlabel('Voltage (V)');
ylabel('Current (A)');
plot(V,I_actual);
plot(V,I_p4);
legend('Actual', 'Poly 4');

subplot(1,2,2);
title('4 Term Polynomial Fit');
hold on;
xlabel('Voltage (V)');
ylabel('Current (A)');
plot(V,I_actual);
plot(V,I_p8);
legend('Actual', 'Poly 8');

% Part 3 - Nonlinear fit
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(V, I_test, fo);
I_f = ff(V);

f_3 = figure('Name', 'Part 3');
hold on;
plot(V,I_actual);
plot(V,I_f);
title('Non-linear Fit');
legend('Actual', 'Non-linear Fit');
xlabel('Voltage (V)');
ylabel('Current (A)');

% Part 4 - Neural Net
inputs = V.';
targets = I_test.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs;

f_4 = figure('Name', 'Part 4');
hold on;
plot(V,I_actual);
plot(V,outputs');
title('Non-linear Fit');
legend('Actual', 'Neural Net Fit');
xlabel('Voltage (V)');
ylabel('Current (A)');

figure(f_3);
figure(f_2);
figure(f_1);