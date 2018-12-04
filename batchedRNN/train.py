from utils import *
from model.RoseSeq2Seq import Seq2Seq
from torch.optim.lr_scheduler import MultiStepLR

def main():
    # Set Additional Args
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args._device = "cuda" if args.cuda else "cpu"
    args.use_attn = not args.no_attn
    args.use_schedule_sampling = not args.no_schedule_sampling
    # Get Data
    if args.local:
        baseDataDir = "/Users/danielzeiberg/Documents/Data"
    else:
        baseDataDir = "/home/dan/data"
    if args.dataset == "traffic":
        dataDir = baseDataDir + "/Traffic/Processed/trafficWithTime/"
    elif args.dataset == "human":
        dataDir = baseDataDir + "/Human/Processed/INPUT_HORIZON_25_PREDICTION_HORIZON_50/"
    else:
        assert False, "bad dataset specified"
    data = getDataLoaders(dataDir)
    # Set data dependent Args
    args.x_dim = data['x_dim']
    args.input_sequence_len = data['input_sequence_len']
    args.target_sequence_len = data["target_sequence_len"]
    args.channels = data["channels"]
    if args.dataset == "traffic":
        args.output_dim = args.x_dim
    else: #args.dataset == "human":
        args.output_dim = args.x_dim * 2
    # Get Save Dir
    args.save_dir = getSaveDir()
    # Save Args
    saveUsefulData()
    # Generate Model
    model = Seq2Seq(args)
    if args.cuda:
        model = model.cuda()
    # Get Training Stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.lambda_l2)
    #lr_decay_milestones = np.arange(args.lr_decay_beginning, args.n_epochs, args.lr_decay_every)
    #scheduler = MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=args.lr_decay_factor)
    # Callable that transforms the data to be ready to use
    # callableTransform = partial(transformBatch, scaler=data["scaler"])
    experimentResults = {
    "train_recon_losses": [],
    "val_recon_losses": []
    }
    EarlyStopper = EarlyStoppingObject()
    for epoch in range(args.n_epochs):
        model.train()
        running_loss = 0.0
        epoch_train_loss = 0.0
        nTrainBatches = 0
        for batchIDX, (inputData, target) in enumerate(map(data["scaler"].transformBatchForEpoch, data["train"]), 1):
            nTrainBatches += 1
            optimizer.zero_grad()
            output = model(inputData, target, epoch)
            loss = getLoss(model, output, target, data["scaler"])
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            if batchIDX % args.print_every == 0:
                logging.info("epoch:{} batch:{} running avg loss: {:.4f}".format(
                    epoch+1,
                    batchIDX,
                    running_loss / args.print_every
                ))
                running_loss = 0.0
        # Validate
        model.eval()
        epoch_val_loss = 0.0
        nValBatches = 0
        with torch.no_grad():
            for batchIDX, (inputData, target) in enumerate(map(data["scaler"].transformBatchForEpoch, data["val"])):
                nValBatches += 1
                output = model(inputData, target, epoch)
                loss = getLoss(model, output, target, data["scaler"])
                epoch_val_loss += loss.item()
        # Print Epoch Results
        logging.info("epoch:{} avg training loss: {:.4f} avg val loss: {:.4f}".format(
            epoch + 1,
            epoch_train_loss / nTrainBatches,
            epoch_val_loss / nValBatches
        ))
        # Store Epoch Results
        experimentResults["train_recon_losses"].append(epoch_train_loss / nTrainBatches)
        experimentResults["val_recon_losses"].append(epoch_val_loss / nValBatches)
        # Check if I should stop early
        if EarlyStopper.checkStop(experimentResults["val_recon_losses"][-1]):
            saveModel(model.state_dict(), epoch)
            break
        # Save Model if necessary
        if (epoch % args.save_freq) == 0:
            saveModel(model.state_dict(), epoch)
    model_fn = args.save_dir + '{}_full_model'.format(args.model) +".pth"
    modelWeights = model.state_dict()
    torch.save(modelWeights, model_fn)
    logging.info('Saved model to '+model_fn)
    plotLosses(experimentResults["train_recon_losses"], experimentResults["val_recon_losses"])
    return experimentResults["train_recon_losses"][-1], experimentResults["val_recon_losses"][-1]

if __name__ == '__main__':
    logging.info('Starting training script')
    main()