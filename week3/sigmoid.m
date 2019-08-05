function g = sigmoid(z)
  enumerator = ones(size(z)); % (20x1)
  denominator = 1 + exp(-z); % (20x1)

  g = enumerator ./ denominator; % (20x1)
end
