format long g;
clear;
clc;

e1 = exp(1);
em1 = exp(-1);

% H5(x) = a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + x + 1
a5 = 1 + em1 - e1 / 2;
a4 = 1 - 3 * em1 / 4 - e1 / 4;
a3 = e1 - 2 - 3 * em1 / 2;
a2 = -2 + 5 * em1 / 4 + 3 * e1 / 4;

H = @(x) (((((a5 .* x + a4) .* x + a3) .* x + a2) .* x + 1) .* x + 1);
Hp = @(x) ((((5 * a5 .* x + 4 * a4) .* x + 3 * a3) .* x + 2 * a2) .* x + 1);
Hpp = @(x) (((20 * a5 .* x + 12 * a4) .* x + 6 * a3) .* x + 2 * a2);

eps_fun = @(x) exp(x) - H(x);
eps_p = @(x) exp(x) - Hp(x);
eps_pp = @(x) exp(x) - Hpp(x);

[root_left, history_left] = newton_solve(eps_p, eps_pp, -0.6, 20, 1e-14);
[root_right, history_right] = newton_solve(eps_p, eps_pp, 0.6, 20, 1e-14);

value_left = eps_fun(root_left);
value_right = eps_fun(root_right);

x = linspace(-1, 1, 1000);
y = eps_fun(x);

fig = figure('Visible', 'off');
plot(x, y, 'b-', 'LineWidth', 1.8);
hold on;
plot(root_left, value_left, 'ro', 'MarkerSize', 7, 'LineWidth', 1.2);
plot(root_right, value_right, 'ro', 'MarkerSize', 7, 'LineWidth', 1.2);
plot([-1, 0, 1], [0, 0, 0], 'ks', 'MarkerSize', 5, 'LineWidth', 1.0);
grid on;
xlabel('x');
ylabel('\epsilon(x) = e^x - H_5(x)');
title('Hermite interpolation error on [-1,1]');
legend('\epsilon(x)', 'Local maxima', 'Interpolation nodes', 'Location', 'northwest');
exportgraphics(fig, 'epsilon_plot.png', 'Resolution', 200);
close(fig);

theoretical_bound = exp(1) / 4860;

fid = fopen('matlab_results.txt', 'w');
fprintf(fid, 'H5 coefficients [a5,a4,a3,a2,a1,a0]:\n');
fprintf(fid, '%.15f %.15f %.15f %.15f %.15f %.15f\n', a5, a4, a3, a2, 1, 1);
fprintf(fid, '\nNewton root in (-1,0): %.15f\n', root_left);
fprintf(fid, 'epsilon(root_left) = %.15e\n', value_left);
fprintf(fid, '\nNewton root in (0,1): %.15f\n', root_right);
fprintf(fid, 'epsilon(root_right) = %.15e\n', value_right);
fprintf(fid, '\nTheoretical upper bound = %.15e\n', theoretical_bound);
fprintf(fid, 'Actual/theoretical ratio = %.15f\n', value_right / theoretical_bound);

fprintf(fid, '\nLeft iteration history:\n');
fprintf(fid, 'k\tx_k\tepsilon''(x_k)\t|x_k-x_{k-1}|\n');
for i = 1:size(history_left, 1)
    fprintf(fid, '%d\t%.15f\t%.15e\t', history_left(i, 1), history_left(i, 2), history_left(i, 3));
    if isnan(history_left(i, 4))
        fprintf(fid, 'NaN\n');
    else
        fprintf(fid, '%.15e\n', history_left(i, 4));
    end
end

fprintf(fid, '\nRight iteration history:\n');
fprintf(fid, 'k\tx_k\tepsilon''(x_k)\t|x_k-x_{k-1}|\n');
for i = 1:size(history_right, 1)
    fprintf(fid, '%d\t%.15f\t%.15e\t', history_right(i, 1), history_right(i, 2), history_right(i, 3));
    if isnan(history_right(i, 4))
        fprintf(fid, 'NaN\n');
    else
        fprintf(fid, '%.15e\n', history_right(i, 4));
    end
end

fclose(fid);

disp('MATLAB analysis finished.');

function [root, history] = newton_solve(f, fp, x0, max_iter, tol)
    history = zeros(max_iter + 1, 4);
    x = x0;
    history(1, :) = [0, x, f(x), NaN];
    for k = 1:max_iter
        fx = f(x);
        fpx = fp(x);
        x_next = x - fx / fpx;
        fx_next = f(x_next);
        history(k + 1, :) = [k, x_next, fx_next, abs(x_next - x)];
        if abs(x_next - x) < tol || abs(fx_next) < tol
            history = history(1:k + 1, :);
            root = x_next;
            return;
        end
        x = x_next;
    end
    history = history(1:max_iter + 1, :);
    root = x;
end
