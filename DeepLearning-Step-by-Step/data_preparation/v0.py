
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_tensor = torch.as_tensor(X_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
