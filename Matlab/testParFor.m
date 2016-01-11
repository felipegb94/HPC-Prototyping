n = 30000;
x = randn(1,n) ;
y = zeros(3,n);
tic
for i = 1 : n
  y(1,i) = std(x(1:i));
end
fprintf('\n For normal: %f secs\n',toc);
tic
for i = drange(1 : n)
  y(2,i) = std(x(1:i));
end
fprintf('\n For drange: %f secs\n',toc);
tic
parfor i = 1 : n
  y(3,i) = std(x(1:i));
end
fprintf('\n parFor: %f secs\n',toc);