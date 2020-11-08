import os
import numpy
import argparse
import time
import torch
import torchaudio
from model.model import Room2Reverb
from model.data_loader import CreateDataLoader


import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="room2reverb", help="Name of the experiment. It decides where to store samples and models.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--encoder_path", type=str, default="resnet50_places365.pth.tar", help="Path to pre-trained Encoder ResNet50 model.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--dataset", type=str, default="room2reverb", help="Name of dataset located in the dataset folder.")
    parser.add_argument("--print_freq", type=int, default=100, help="Frequency of showing training results on console.")
    parser.add_argument("--niter", type=int, default=200, help="Number of training iters.")
    parser.add_argument("--save_latest_freq", type=int, default=1000, help="Frequency of saving the latest results.")
    parser.add_argument("--save_epoch_freq", type=int, default=10, help="Frequency of saving checkpoints at end of epochs.")
    parser.add_argument("--resize_or_crop", type=str, default="scale_width_and_crop", help="Scaling and cropping of images at load time.")
    args = parser.parse_args()

    print_freq = lcm(args.print_freq, args.batch_size)
    args.batchSize = args.batch_size
    args.serial_batches = False
    args.nThreads = 2
    args.max_dataset_size = float("inf")
    args.dataroot = os.path.join("./datasets", args.name)
    args.phase = "train"
    args.isTrain = True
    args.loadSize = 512
    args.fineSize = 224
    args.no_flip = True
    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("#training images = %d" % dataset_size)

    # Model dir
    folder = args.checkpoints_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Main model
    model = Room2Reverb(args.encoder_path)

    # Train settings
    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    print_delta = total_steps % args.print_freq
    save_delta = total_steps % args.save_latest_freq

    # Train model
    for epoch in range(start_epoch, args.niter + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % args.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += args.batch_size
            epoch_iter += args.batch_size

            # Model training
            label = data["label"].cuda()
            spec = data["image"].cuda()
            model.train_step(spec, label, (i % 3) == 0, (i % 3 == 0))

            # Print progress
            if (total_steps % args.print_freq) == print_delta:
                t = (time.time() - iter_start_time) / args.print_freq
                message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, i, t)
                message += "G: %.3f D: %.3f " % (model.G_loss.item(), model.D_loss.item())
                print(message)

            # Store model
            if total_steps % args.save_latest_freq == save_delta:
                print("saving the latest model (epoch %d, total_steps %d)" % (epoch, total_steps))
                save_network(model.g, "G", "latest", folder)
                save_network(model.d, "D", "latest", folder)

            if epoch_iter >= dataset_size:
                break

        print("End of epoch %d / %d \t." % (epoch, args.niter))

        # Save model
        if epoch % args.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d, iters %d" % (epoch, total_steps))
            save_network(model.g, "G", "latest", folder)
            save_network(model.d, "D", "latest", folder)
            save_network(model.g, "G", epoch, folder)
            save_network(model.d, "D", epoch, folder)



def save_network(network, network_label, epoch_label, save_dir):
    """Store model."""
    save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.cuda()



if __name__ == "__main__":
    main()
