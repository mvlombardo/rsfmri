function e = msentropy(input, m, r, factor)
%   msentropy - calculate multi-scale entropy
%
%   INPUT
%       input = time-series data as input
%       m = m parameter (e.g., 2)
%       r = r parameter (e.g., 0.15)
%       factor = scale factor (e.g., 5)
%
%   Example usage
%
%   m = 2; r = 0.15; factor = 5;
%   e = msentropy(parcel_timeseries, m, r, factor);
%

y = input;
y = y - mean(y);
y =y/std(y);

for i = 1:factor
   s = coarsegraining(y,i);
   sampe = sampenc(s,m+1,r);
   e(i) = sampe(m+1);   
end
e = e';

end % function msentropy

%% coarsegraining function
function output = coarsegraining(input, factor)

n = length(input);
blk = fix(n/factor);
for i = 1:blk
   s(i) = 0; 
   for j = 1:factor
      s(i) = s(i) + input(j + (i - 1)*factor);
   end    
   s(i) = s(i)/factor;
end
output = s';

end % function coarsegraining

%% sampenc function
function [e, A, B] = sampenc(y, M, r)
%function [e, A, B] = sampenc(y, M, r);
%
%   Input
%
%       y input data
%       M maximum template length
%       r matching tolerance
%
%   Output
%
%       e sample entropy estimates for m=0,1,...,M-1
%       A number of matches for m=1,...,M
%       B number of matches for m=0,...,M-1 excluding last point

n = length(y);
lastrun = zeros(1,n);
run = zeros(1,n);
A = zeros(M,1);
B = zeros(M,1);
p = zeros(M,1);
e = zeros(M,1);
for i = 1:(n-1)
   nj = n-i;
   y1 = y(i);
   for jj = 1:nj
      j = jj+i;      
      if abs(y(j)-y1)<r
         run(jj) = lastrun(jj)+1;
         M1 = min(M,run(jj));
         for m = 1:M1           
            A(m) = A(m)+1;
            if j<n
               B(m) = B(m)+1;
            end % if j<n
         end % for m = 1:M1
      else
         run(jj) = 0;
      end % if abs(y(j)-y1)<r
   end % for jj = 1:nj
   for j = 1:nj
      lastrun(j) = run(j);
   end % for j = 1:nj
end % for i = 1:(n-1)

N = n*(n-1)/2;
B = [N;B(1:(M-1))];
p = A./B;
e = -log(p);

end % function sampenc
