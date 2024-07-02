clear all; clc;


an11 = -0.5;
an20 = 0;
an10 = 2.88;
bn11 = -0.5;
bn20 = 0;
bn10 = 2.88;
k = 1;
nu = 0.34023;
awo = 2;
bwo = 2;
range = 3;
aN1 = @(ax1, ay1) an11*(ax1^2+ay1^2)+an10;
aN2 = @(ax2, ay2) an20;
bN1 = @(bx1, by1) bn11*(bx1^2+by1^2)+bn10;
bN2 = @(bx2, by2) bn20;

dax1dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -awo*ay1+aN1(ax1, ay1)*ax1-k*ay2;
day1dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) awo*ax1+aN1(ax1, ay1)*ay1+k*ax2;
dax2dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -awo*ay2+aN2(ax2, ay2)*ax2-k*ay1-nu*by1;
day2dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) awo*ax2+aN2(ax2, ay2)*ay2+k*ax1+nu*bx1;

dbx1dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -bwo*by1+bN1(bx1, by1)*bx1-k*by2-nu*ay2;
dby1dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) bwo*bx1+bN1(bx1, by1)*by1+k*bx2+nu*ax2;
dbx2dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -bwo*by2+bN2(bx2, by2)*bx2-k*by1;
dby2dt = @(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) bwo*bx2+bN2(bx2, by2)*by2+k*bx1;


sys = @(t, y) [dax1dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               day1dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               dax2dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               day2dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               dbx1dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               dby1dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               dbx2dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8));
               dby2dt(y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8))];

%initial_conditions = [range*rand, range*rand, range*rand, range*rand, range*rand, range*rand, range*rand, range*rand];
initial_conditions = [1, 1, 1, 1, 1, 1, 1, 1];
while any(initial_conditions == 0)
    initial_conditions = [range*rand, range*rand, range*rand, range*rand, range*rand, range*rand, range*rand, range*rand];
end

tic;
options = odeset('RelTol',1e-6,'AbsTol',1e-8, 'Events', @eventsFcn);
tspan = [0, 100000];
[t_sol, Y_sol, te, ye, ie] = ode45(sys, tspan, initial_conditions, options);
disp("stage 1 time: " + toc);

if ~isempty(te)
    error("Unbounded solution.")
end

initial_conditions = [Y_sol(end, :)];

tic;
tspan = [0, 10000];
options = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_sol, Y_sol] = ode45(sys, tspan, initial_conditions, options);
disp("stage 2 time: " + toc);

index_start = round((1 / 2) * length(t_sol));
tsol = t_sol(index_start:end);
dt = 0.005;
t_start = tsol(1);                   
t_end = tsol(end);                   
tsol_const = t_start:dt:t_end; 
sol_const_col1 = interp1(tsol, Y_sol(index_start:end, 1), tsol_const);
sol_const_col2 = interp1(tsol, Y_sol(index_start:end, 2), tsol_const);
sol_const_col3 = interp1(tsol, Y_sol(index_start:end, 3), tsol_const);
sol_const_col4 = interp1(tsol, Y_sol(index_start:end, 4), tsol_const);
sol_const_col5 = interp1(tsol, Y_sol(index_start:end, 5), tsol_const);
sol_const_col6 = interp1(tsol, Y_sol(index_start:end, 6), tsol_const);
sol_const_col7 = interp1(tsol, Y_sol(index_start:end, 7), tsol_const);
sol_const_col8 = interp1(tsol, Y_sol(index_start:end, 8), tsol_const);
sol_const_combined = [sol_const_col1; sol_const_col2; sol_const_col3; sol_const_col4; sol_const_col5; sol_const_col6; sol_const_col7; sol_const_col8]';
index_cutoff = find(tsol_const == t_start);
cutoff = sol_const_combined(index_cutoff, :);
eps = 0.05;


cutoff_indices = find(all(abs(bsxfun(@minus, sol_const_combined, cutoff)) < eps, 2));  
cutoff_times = tsol_const(cutoff_indices); 
index_found = false;
for i = 2 : length(cutoff_indices)
    if cutoff_times(i) - cutoff_times(1) > 3000
        index_found = true;
        disp("Repetition found.")
        repeat_index = cutoff_indices(i);
        break
    end
end

if index_found == false
    cutoff_times = 1;
    repeat_index = length(tsol_const);
    disp("No repetition found.")
end


disp("Time of signal being transformed: " + (tsol_const(repeat_index) - cutoff_times(1)))
sol_const = sol_const_combined(cutoff_indices(1) : repeat_index, 2);
tsol_const = tsol_const(cutoff_indices(1) : repeat_index);

figure;
plot(tsol_const, sol_const);
xlabel('$t$', 'Interpreter','latex');
ylabel('$Im[a_1]$', 'Interpreter','latex');



FT_Y2 = fft(sol_const);
FT_Y2 = FT_Y2 / length(sol_const);


power_Y2 = abs(FT_Y2).^2;
N = length(tsol_const);   
T = max(tsol_const) - min(tsol_const);
frequencies = (0:N-1)/T;  


figure;
semilogy(frequencies, power_Y2);
title('$Im[a_1]$', 'Interpreter', 'latex');
xlabel('Frequency (Hz)');
ylabel('Power');

N = 10;
range_limit = round(length(power_Y2) * 0.5);
[peaks, locs] = findpeaks(power_Y2(1:range_limit));
[sorted_peaks, sorted_idxs] = sort(peaks, 'descend');
top_peaks = zeros(1, N);
top_locs = zeros(1, N);
count = 1;
for i = 1:length(sorted_peaks)
    if ~ismember(locs(sorted_idxs(i)), top_locs)
        top_peaks(count) = sorted_peaks(i);
        top_locs(count) = locs(sorted_idxs(i));
        count = count + 1;
    end

    if count > N
        break;
    end
end
top_freqs = frequencies(top_locs);
[max_peak, idx_max_peak] = max(peaks);

if top_peaks(1) < 0.1
     cost = inf;
else
     cost = top_peaks(1)/top_peaks(2);
end
disp("FFT ratio: " + cost)





%%%%%%%

function [value, isterminal, direction] = eventsFcn(t, y)
bound = 1e4;
value = max(abs(y)) - bound; % Event function triggers when the max absolute value of y exceeds the bound
isterminal = 1;
direction = 0;
end











