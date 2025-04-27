EPOCHS = 200

losses = []
val_losses = []

for epoch in range(EPOCHS):
    loss = mini_batch_fn(device, train_loader, train_step_fn)
    losses.append(loss)
    with torch.inference_mode():
        val_loss = mini_batch_fn(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    writer.add_scalars(
        main_tag='loss',
        tag_scalar_dict={
            'training': loss,
            'validation': val_loss
        },
        global_step=epoch
    )
writer.close()
