function DowJonesNewscorncarryfwdalldictssentiment1 = load_sentiment(filename, dataLines)
%IMPORTFILE Import data from a text file
%  DOWJONESNEWSCORNCARRYFWDALLDICTSSENTIMENT1 = IMPORTFILE(FILENAME)
%  reads data from text file FILENAME for the default selection.
%  Returns the data as a table.
%
%  DOWJONESNEWSCORNCARRYFWDALLDICTSSENTIMENT1 = IMPORTFILE(FILE,
%  DATALINES) reads data for the specified row interval(s) of text file
%  FILENAME. Specify DATALINES as a positive scalar integer or a N-by-2
%  array of positive scalar integers for dis-contiguous row intervals.
%
%  Example:
%  DowJonesNewscorncarryfwdalldictssentiment1 = importfile("./cbot_data/DowJonesNews_corn_carry_fwd_alldictssentiment.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 19-Aug-2022 16:32:11

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Dates", "customdict", "usdacmecftc", "usdasplitparenth"];
opts.VariableTypes = ["datetime", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Dates", "InputFormat", "yyyy-MM-dd");

% Import the data
DowJonesNewscorncarryfwdalldictssentiment1 = readtable(filename, opts);

end