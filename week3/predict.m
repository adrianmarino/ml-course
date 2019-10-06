function p = predict(theta, X)
  %        (mx3)  (3x1)
  h_theta =  X  * theta; % (mx1)
  p = sigmoid(h_theta) >= 0.5 % (mx1)
end