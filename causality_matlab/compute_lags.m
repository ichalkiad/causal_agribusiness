function [input_to_lagged, input_from_lagged, input_side_lagged] = ...
                                        compute_lags(Data, to, from, side, lag)

input_to_lagged   = [];
input_from_lagged = [];
input_side_lagged = [];
a = lagmatrix(Data(:,[to]),1:lag);
input_to_lagged = a(lag+1:end,:);

if length(from) > 0
    b = lagmatrix(Data(:,[from]),1:lag);
    input_from_lagged = b(lag+1:end,:);
end

if length(side) > 0
    c = lagmatrix(Data(:,[side]),1:lag);
    input_side_lagged = c(lag+1:end,:);
end

end