import torch

def get_batch(split, data, block_size, batch_size):
    """
    Retrieves a batch of data for training or evaluation.

    Args:
    - split (str): The type of data ('train' or 'val') to fetch the batch from.
    - data (Tensor): The dataset, typically a tensor containing the training or validation data.
    - block_size (int): The number of sequential data points to include in each training example (sequence length).
    - batch_size (int): The number of examples in a single batch.

    Returns:
    - x_batch (Tensor): Input batch tensor of shape (batch_size, block_size), which contains the features.
    - y_batch (Tensor): Target batch tensor of shape (batch_size, block_size), which contains the labels (next token predictions).
    
    This function randomly selects starting points within the data and creates input-output pairs (x_batch, y_batch)
    for model training or evaluation.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_batch = torch.stack([data[i:i + block_size] for i in ix])
    y_batch = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x_batch, y_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, config):
    """
    Estimates the training and validation loss over a number of evaluation iterations.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - train_data (Tensor): The training dataset.
    - val_data (Tensor): The validation dataset.
    - eval_iters (int): The number of evaluation iterations to perform.
    - config (dict): A configuration dictionary containing `block_size` and `batch_size`.

    Returns:
    - out (dict): A dictionary containing the average loss for the 'train' and 'val' datasets.

    This function switches the model to evaluation mode, computes the loss on batches of data for both 
    training and validation datasets, averages the results over multiple evaluation iterations, and returns
    the mean loss for each split.
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split, data, config['block_size'], config['batch_size'])
            _, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, train_data, val_data, optimizer, config):
    """
    Trains the model over a specified number of iterations, periodically evaluating it on training and validation data.

    Args:
    - model (torch.nn.Module): The model to train.
    - train_data (Tensor): The training dataset.
    - val_data (Tensor): The validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - config (dict): A configuration dictionary containing `max_iterations`, `eval_interval`, 
                     `block_size`, `batch_size`, and `eval_iters`.

    The function performs training in an iterative loop. After every `eval_interval` iterations, it estimates
    the loss on both the training and validation datasets using the `estimate_loss` function. It also performs 
    backpropagation and updates the model's weights after every batch.
    
    The training process continues until `max_iterations` is reached.
    """
    for iteration in range(config['max_iterations']):
        if iteration % config['eval_interval'] == 0 or iteration == config['max_iterations'] - 1:
            losses = estimate_loss(
                model, train_data, val_data, config['eval_iters'], config
            )
            print(f"step {iteration}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        x_batch, y_batch = get_batch('train', train_data, config['block_size'], config['batch_size'])
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
