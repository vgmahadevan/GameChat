import os
import torch
import config
import matplotlib.pyplot as plt
from model_utils import ModelDefinition
from models import FCNet, BarrierNet
from data_logger import DataGenerator, Dataset
from sklearn.model_selection import train_test_split

def train(dataloader, model, loss_fn, optimizer, losses):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(config.device), y.to(config.device)
        
        # Compute prediction error
        pred = model(X, 1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:  #25
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test(dataloader, model, loss_fn, losses):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.device), y.to(config.device)
            pred = model(X, 1)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses

if __name__ == "__main__":
    params = {'batch_size': config.train_batch_size,
            'shuffle': True,
            'num_workers': 4}

    generator = DataGenerator(config.train_data_paths, config.x_is_d_goal)

    norm_inputs, input_mean, input_std = generator.get_inputs(agent_idx=config.agent_to_train, normalize=True)
    norm_outputs, output_mean, output_std = generator.get_outputs(agent_idx=config.agent_to_train, normalize=True)

    X_train, X_test, y_train, y_test = train_test_split(norm_inputs, norm_outputs, test_size=0.25, random_state=42, shuffle=True)

    print("Train size:", len(X_train), "Test size:", len(X_test))

    # Generators
    training_set = Dataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(training_set, **params)

    testing_set = Dataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(testing_set, **params)

    model_definition = ModelDefinition(
        is_barriernet=config.use_barriernet,
        weights_path=None,
        nHidden1=config.nHidden1,
        nHidden21=config.nHidden21,
        nHidden22=config.nHidden22,
        nHidden23=config.nHidden23 if config.add_control_limits else None,
        input_mean=input_mean.tolist(),
        input_std=input_std.tolist(),
        label_mean=output_mean.tolist(),
        label_std=output_std.tolist(),
        add_control_limits=config.add_control_limits,
        add_liveness_filter=config.add_liveness_filter,
        separate_penalty_for_opp=config.separate_penalty_for_opp,
        x_is_d_goal=config.x_is_d_goal
    )

    if config.use_barriernet:
        model = BarrierNet(model_definition, generator.get_obstacles(), generator.data_streams[0]["iterations"][0]["goals"][config.agent_to_train]).to(config.device)
    else:
        model = FCNet(model_definition).to(config.device)
    print(model_definition)
    print(model)

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_losses, test_losses = [], []
    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses)
        print("Finished training epoch")
        test_losses = test(test_dataloader, model, loss_fn, test_losses)
    print("Training Done!")

    # Save the model.
    saveprefix = config.saveprefix + ('_bn' if config.use_barriernet else '_fc')
    weights_path = saveprefix + '.pth'
    torch.save(model.state_dict(), weights_path)
    model_definition.weights_path = os.path.basename(weights_path)
    model_definition.save(saveprefix + '_definition.json')
    print(f"Saved PyTorch Model and Definition to {saveprefix}")

    model.eval()    
    tr = []
    ctrl1, ctrl2, ctrl1_real, ctrl2_real = [], [], [], []
    t0 = 0

    with torch.no_grad():
        for X, y in zip(X_test, y_test):
            x = torch.autograd.Variable(torch.from_numpy(X), requires_grad=False)
            x = torch.reshape(x, (1, len(X)))
            x = x.to(config.device)
            ctrl = model(x, 0)
            
            if config.use_barriernet:
                ctrl1.append(ctrl[0])
                ctrl2.append(ctrl[1])
            else:
                ctrl1.append(ctrl[0,0].item())
                ctrl2.append(ctrl[0,1].item())
            ctrl1_real.append(y[0])
            ctrl2_real.append(y[1])
            tr.append(t0)
            t0 = t0 + 0.2

    print("Test done!")    

    plt.figure(1)
    plt.plot(tr, ctrl1_real, color = 'red', label = 'actual(optimal)')
    plt.plot(tr, ctrl1, color = 'blue', label = 'implemented')
    plt.legend()
    plt.ylabel('Angular speed (control)')
    plt.xlabel('time')

    plt.savefig('train_results/angular_speed_control.pdf')

    plt.figure(2)
    plt.plot(tr, ctrl2_real, color = 'red', label = 'actual(optimal)')
    plt.plot(tr, ctrl2, color = 'blue', label = 'implemented')
    plt.legend()
    plt.ylabel('Acceleration (control)')
    plt.xlabel('time')

    plt.savefig('train_results/acceleration_control.pdf')

    plt.figure(3)    
    plt.title('Train Loss')
    plt.plot(train_losses, color = 'green', label = 'train')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('time')
    plt.ylim(ymin=0.)

    plt.savefig('train_results/train_loss.pdf')

    plt.figure(4)
    plt.title('Test Loss')
    plt.plot(test_losses, color = 'red', label = 'test')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('time')
    plt.ylim(ymin=0.)

    plt.savefig('train_results/test_loss.pdf')
