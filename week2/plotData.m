function plotData(x, y)
  figure;
  plot(x,  y,  'rx',  'MarkerSize',  15);
  ylabel('Profit (in $10,000s)');
  xlabel('City population (in 10,000s)');
end
