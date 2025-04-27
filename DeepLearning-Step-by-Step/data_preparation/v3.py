
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LR = 0.1
torch.manual_seed(SEED)
model = nn.Sequential(
    nn.Linear(1, 1)
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)

criterion = nn.MSELoss(reduction='mean')

train_step_fn = make_train_step_fn(model, criterion, optimizer)

val_step_fn = make_val_step_fn(model, criterion)

dummy_X, dummy_y = see_loader_fn(train_loader)

writer.add_graph(model, dummy_X.to(device))
