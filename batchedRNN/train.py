from utils import *
from model.RoseSeq2Seq import Seq2Seq
from model.SketchRNN import SketchRNN
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
        if args.debugDataset:
            dataDir += "down_sample_0.1/"
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
    if args.model == "rnn":
        model = Seq2Seq(args)
    elif args.model == "sketch-rnn":
        model = SketchRNN(args)
    else:
        assert False, "bad model specified"
    if args.cuda:
        model = model.cuda()
    # Get Training Stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.lambda_l2)
    #lr_decay_milestones = np.arange(args.lr_decay_beginning, args.n_epochs, args.lr_decay_every)
    #scheduler = MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=args.lr_decay_factor)
    # Callable that transforms the data to be ready to use
    experimentResults = {
    "train_recon_losses": [],
    "train_kld_losses" : [],
    "train_total_losses" : [],
    "val_recon_losses": [],
    "val_kld_losses": [],
    "val_total_losses": []
    }
    EarlyStopper = EarlyStoppingObject()
    for epoch in range(args.n_epochs):
        model.train()
        running_recon_loss = 0.0
        running_kld_loss = 0.0
        epoch_train_recon_loss = 0.0
        epoch_train_kld_loss = 0.0
        currentKLDWeight = getKLDWeight(epoch)
        nTrainBatches = 0
        for batchIDX, (inputData, target) in enumerate(map(data["scaler"].transformBatchForEpoch, data["train"]), 1):
            nTrainBatches += 1
            optimizer.zero_grad()
            output = model(inputData, target, epoch)
            reconLoss, kldLoss = getLoss(model, output, target, data["scaler"], epoch)
            loss = reconLoss + kldLoss * currentKLDWeight
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # print statistics
            running_recon_loss += reconLoss.item()
            epoch_train_recon_loss += reconLoss.item()
            if args.model == "sketch-rnn":
                running_kld_loss += kldLoss.item() * currentKLDWeight
                epoch_train_kld_loss += kldLoss.item() * currentKLDWeight
            if batchIDX % args.print_every == 0:
                logging.info("epoch:{} batch:{} running avg losses: recon loss: {:.4f} kld loss: {:.4f} total loss: {:.4f}".format(
                    epoch+1,
                    batchIDX,
                    running_recon_loss / args.print_every,
                    running_kld_loss / args.print_every,
                    (running_recon_loss + running_kld_loss) / args.print_every
                ))
                running_recon_loss = 0.0
                running_kld_loss = 0.0
        # Validate
        model.eval()
        epoch_val_recon_loss = 0.0
        epoch_val_kld_loss = 0.0
        nValBatches = 0
        with torch.no_grad():
            for batchIDX, (inputData, target) in enumerate(map(data["scaler"].transformBatchForEpoch, data["val"])):
                nValBatches += 1
                output = model(inputData, target, epoch)
                reconLoss, kldLoss = getLoss(model, output, target, data["scaler"], epoch)
                epoch_val_recon_loss += reconLoss.item()
                if args.model == "sketch-rnn":
                    epoch_val_kld_loss += kldLoss.item() * currentKLDWeight
        # Print Epoch Results
        logging.info("epoch:{} avg losses: training recon: {:.4f} train kld: {:.4f} val recon: {:.4f} val kld: {:.4f} total train: {:.4f} total val: {:.4f}".format(
            epoch + 1,
            epoch_train_recon_loss / nTrainBatches,
            epoch_train_kld_loss / nTrainBatches,
            epoch_val_recon_loss / nValBatches,
            epoch_val_kld_loss / nValBatches,
            (epoch_train_recon_loss + epoch_train_kld_loss) / nTrainBatches,
            (epoch_val_recon_loss + epoch_val_kld_loss) / nValBatches
        ))
        # Store Epoch Results
        experimentResults["train_recon_losses"].append(epoch_train_recon_loss / nTrainBatches)
        experimentResults["train_kld_losses"].append(epoch_train_kld_loss / nTrainBatches)
        experimentResults["train_total_losses"].append((epoch_train_recon_loss + epoch_train_kld_loss) / nTrainBatches)
        experimentResults["val_recon_losses"].append(epoch_val_recon_loss / nValBatches)
        experimentResults["val_kld_losses"].append(epoch_val_kld_loss / nValBatches)
        experimentResults["val_total_losses"].append((epoch_val_recon_loss + epoch_val_kld_loss) / nValBatches)
        # Check if I should stop early
        if EarlyStopper.checkStop(experimentResults["val_total_losses"][-1]):
            saveModel(model.state_dict(), epoch)
            break
        # Save Model if necessary
        if (epoch % args.save_freq) == 0:
            saveModel(model.state_dict(), epoch)
    model_fn = args.save_dir + '{}_full_model'.format(args.model) +".pth"
    modelWeights = model.state_dict()
    torch.save(modelWeights, model_fn)
    logging.info('Saved model to '+model_fn)
    torch.save(experimentResults["train_recon_losses"], args.save_dir+"plot_train_recon_losses")
    torch.save(experimentResults["val_recon_losses"], args.save_dir+"plot_val_recon_losses")
    torch.save(experimentResults["train_kld_losses"], args.save_dir+"plot_train_KLD_losses")
    torch.save(experimentResults["val_kld_losses"], args.save_dir+"plot_val_KLD_losses")
    return experimentResults["train_recon_losses"][-1], experimentResults["val_recon_losses"][-1], experimentResults["train_kld_losses"][-1], experimentResults["val_kld_losses"][-1]

if __name__ == '__main__':
    logging.info('Starting training script')
    main()