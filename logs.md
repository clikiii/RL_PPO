
About model names:
origin: no clip for action, smooth l1 for critic
new: lr: 1e-4 -> 5e-4; mse loss for critic
2new: action clip 0.5; green penalty
3new: numtonan(no a good practice)
4new: action clip 0.8 0.2 (fail)
