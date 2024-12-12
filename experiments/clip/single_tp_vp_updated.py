from tqdm import tqdm
import argparse
import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import clip

import sys
sys.path.append(".")
from data import prepare_additive_data
from tools.misc import *
from tools.clip import get_saparate_text_embedding, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from models import AdditiveVisualPrompt
from cfg import *

if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help="Random seed")
    parser.add_argument('--dataset', 
                        choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], 
                        required=True, help="Dataset to use")
    parser.add_argument('--template-number', type=int, default=0, help="Template number for text embeddings")
    parser.add_argument('--epoch', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=40, help="Learning rate")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(args.seed)

    # Experiment setup
    exp = "clip/single_tp_vp"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    os.makedirs(save_path, exist_ok=True)

    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    # Prepare dataset and text embeddings
    loaders, class_names = prepare_additive_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
    templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
    txt_emb = get_saparate_text_embedding(class_names, templates[args.template_number], model)

    # Define network
    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        return logits

    # Visual Prompt Module
    visual_prompt = AdditiveVisualPrompt(224, 30).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(visual_prompt.parameters(), lr=args.lr, momentum=0.9)
    t_max = args.epoch * len(loaders['train'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Logger
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Training setup
    best_acc = 0.0
    scaler = GradScaler()

    for epoch in range(args.epoch):
        visual_prompt.train()
        total_num, true_num, loss_sum = 0, 0, 0

        # Training loop
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                fx = network(visual_prompt(x))
                loss = F.cross_entropy(fx, y, reduction='mean')

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Clamp logit scale
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

            total_num += y.size(0)
            true_num += torch.argmax(fx, dim=1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)

            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")

        # Log training metrics
        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # Scheduler step
        scheduler.step()

        # Testing loop
        visual_prompt.eval()
        total_num, true_num = 0, 0
        with torch.no_grad():
            pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                fx = network(visual_prompt(x))
                total_num += y.size(0)
                true_num += torch.argmax(fx, dim=1).eq(y).sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")

        logger.add_scalar("test/acc", acc, epoch)

        # Save model checkpoints
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }

        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))

        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))

    # Print results
    print("#### -------------------- RESULTS -------------------- ####")
    print("Number of epochs:", args.epoch)
    print("Best Accuracy:", best_acc)
    print("#### -------------------- END OF RESULT -------------------- ####")