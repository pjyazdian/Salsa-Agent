import os
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer
from models.mllm import MotionLLM
from models.training_utils import process_batch_Salsa
from options.option_llm import get_args_parser
from torch.utils.data import DataLoader
from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ---- %{(100*(trainable_params/total_params)):.2f}")

PAIR2LEVEL = {
    f"pair{i}": level
    for i, level in zip(range(1, 10), ["beginner", "intermediate", "professional"] * 3)
}

def collate_fn_salsa(batch):
    """Custom collate function for Salsa_Dataset that handles variable-length tensors.
    
    Returns variable-length tensors (vq_tokens, audio_tokens) as lists, not stacked tensors.
    process_batch_Salsa will handle padding when building prompts.
    """
    # Unpack batch: (level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_info, interhuman_data)
    levels = [item[0] for item in batch]
    ms_desc_L_list = [item[1] for item in batch]
    ms_des_F_list = [item[2] for item in batch]
    vq_tokens_L_list = [item[3] for item in batch]  # Keep as list (variable length)
    vq_tokens_F_list = [item[4] for item in batch]  # Keep as list (variable length)
    audio_tokens_list = [item[5] for item in batch]  # Keep as list (variable length)
    aux_info_list = [item[6] for item in batch]
    interhuman_data_list = [item[7] if len(item) > 7 else None for item in batch]
    
    # Return as tuple (same format as single sample, but batched)
    return (levels, ms_desc_L_list, ms_des_F_list, vq_tokens_L_list, vq_tokens_F_list, 
            audio_tokens_list, aux_info_list, interhuman_data_list)

def train(model, train_loader, args):
    model.train()

    if getattr(model, 'use_gpt_ablation', False):
        model.llm.print_trainable_parameters()
        # Train GPT only (no unused HumanML3D VQ module on this path).
        optimizer = torch.optim.AdamW(model.llm.parameters(), lr=args.lr)
    else:
        model.llm.set_adapter('t2m')  # activate text-to-motion adapter
        #Todo Clone 't2m' adapter weights into 't2m-salsa' for continued fine-tuning
        # model.llm.add_adapter('t2m-salsa', model.lora_config_t2m)
        # model.llm.adapters['t2m-salsa'].load_state_dict(
        #     model.llm.adapters['t2m'].state_dict())

        # Switch to the new adapter
        # model.llm.set_adapter('t2m-salsa')
        model.llm.print_trainable_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # caption, motion = batch[0], batch[1]  # assumes your custom DataLoader outputs this

            # caption, ms_desc_bins, audio_tokens, motion_tokens = batch

            if len(batch) == 8:
                level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch, batch_interhuman = batch
            else:
                level, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch = batch
                batch_interhuman = None

            loss, acc, _, _ = model(level,
                                    ms_desc_L, ms_des_F,
                                    vq_tokens_L, vq_tokens_F, audio_tokens,
                                    batch_interhuman_data=batch_interhuman)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_acc += acc

            if args.use_wandb:
                wandb.log({"batch_loss": loss.item(), "batch_acc": acc})

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        if args.use_wandb:
            wandb.log({"epoch_loss": avg_loss, "epoch_acc": avg_acc})

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch+1)==args.epochs:
            save_path = os.path.join(args.save_dir, f"Xmotionllm_epoch{epoch+1}.pth")
            model.save_model(save_path)
            print(f"Saved checkpoint to {save_path}")


def main():
    args = get_args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MotionLLM uses --learning-rate (llm_lr); default 1e-5 to match Motion-Agent
    args.lr = getattr(args, 'llm_lr', 1e-5)
    args.wandb_project = getattr(args, 'wandb_project', "Salsa-LLM")
    # Stage 1: task 'none' or 'all' = all tasks; use pretrain_all unless --wandb-run-name is set
    _all_tasks = (args.task in (None, 'none', 'all'))
    from models.motion_gpt_ablation import is_gpt_ablation_backbone
    _gpt_ablation = is_gpt_ablation_backbone(getattr(args, 'llm_backbone', None))
    if args.wandb_run_name is None:
        if _gpt_ablation and not _all_tasks:
            args.wandb_run_name = f"gpt_ablation_{args.task}"
        else:
            args.wandb_run_name = "pretrain_all" if _all_tasks else f"{args.task}_v3"
    args.save_dir = getattr(args, 'save_dir', None) or f'output_trained/{args.wandb_run_name}'
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Cache path: is_MDM=True -> lmdb_dir + cache_suffix + '_MDM' (no MotionScript); is_MDM=False -> + cache_suffix only (with MotionScript)
    # Default MDM (backward compat); pass --no-MDM to use non-MDM cache and train with MotionScript data
    args.is_MDM = not getattr(args, 'no_MDM', False)
    # Token set: add MotionScript tokens only when training with MotionScript data (non-MDM); audio from --include-audio
    args.include_motionscript = not args.is_MDM
    if _gpt_ablation:
        # Option A: text-free non-LLM control — never add MotionScript tokens
        args.include_motionscript = False
        if getattr(args, 'motion_repr_type', 'humanml3d') != 'interhuman':
            raise ValueError("gpt_ablation requires --motion-repr-type interhuman")
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        ckpt_config = MotionLLM.load_config_from_checkpoint(args.resume_ckpt)
        args.include_audio = ckpt_config.get('include_audio', args.include_audio)
        args.include_motionscript = ckpt_config.get('include_motionscript', args.include_motionscript)
        if ckpt_config.get('backbone') == 'gpt_ablation':
            args.llm_backbone = 'gpt_ablation'
            args.include_motionscript = False
    lmdb_dir = getattr(args, 'lmdb_dir', 'dataset_processed_New/lmdb_Salsa_pair/lmdb_train')
    # n_poses, subdivision_stride, pose_resampling_fps align with demo.py and README cache creation
    n_poses, subdivision_stride, pose_resampling_fps = 100, 50, 20

    model = MotionLLM(args)
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        model.load_model(args.resume_ckpt)

    model.to(args.device)

    train_dataset = Salsa_Dataset(args,
                    lmdb_dir=lmdb_dir,
                    n_poses=n_poses,
                    subdivision_stride=subdivision_stride,
                    pose_resampling_fps=pose_resampling_fps)
    batch_size = getattr(args, 'train_batch_size', 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_salsa)
    train(model, train_loader, args)


def main2():
    """Minimal test: load dataloader, fetch batch, build prompts via process_batch_Salsa (tokenizer only).
    No MotionLLM/VQVAE loading — for verifying data flow and prompt correctness without GPU memory."""
    args = get_args_parser()
    args.device = torch.device("cpu")
    args.is_MDM = not getattr(args, 'no_MDM', False)
    lmdb_dir = getattr(args, 'lmdb_dir', 'dataset_processed_New/lmdb_Salsa_pair/lmdb_train')
    n_poses, subdivision_stride, pose_resampling_fps = 100, 50, 20
    batch_size = getattr(args, 'train_batch_size', 4)

    train_dataset = Salsa_Dataset(
        args, lmdb_dir=lmdb_dir, n_poses=n_poses,
        subdivision_stride=subdivision_stride, pose_resampling_fps=pose_resampling_fps,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn_salsa,
    )

    batch = next(iter(train_loader))
    levels, ms_desc_L, ms_des_F, vq_tokens_L, vq_tokens_F, audio_tokens, aux_batch, batch_interhuman = batch

    print("--- Batch shapes ---")
    print(f"levels: {levels}")
    print(f"ms_desc_L[0]: {ms_desc_L[0][:200]}..." if len(ms_desc_L[0]) > 200 else f"ms_desc_L[0]: {ms_desc_L[0]}")
    print(f"ms_des_F[0]: {ms_des_F[0][:200]}..." if len(ms_des_F[0]) > 200 else f"ms_des_F[0]: {ms_des_F[0]}")
    for i, (vl, vf, aa) in enumerate(zip(vq_tokens_L, vq_tokens_F, audio_tokens)):
        print(f"  vq_L[{i}].shape={vl.shape}, vq_F[{i}].shape={vf.shape}, audio[{i}].shape={aa.shape}")
    print(f"aux_batch[0] keys: {list(aux_batch[0].keys()) if isinstance(aux_batch[0], dict) else 'n/a'}")
    ih0 = batch_interhuman[0] if batch_interhuman else None
    print(f"batch_interhuman: {type(ih0).__name__ if ih0 is not None else 'None'}")

    tokenizer = AutoTokenizer.from_pretrained(getattr(args, 'llm_backbone', 'google/gemma-2-2b-it'))
    current_task = None if (getattr(args, 'task', None) in (None, 'none', 'all')) else args.task
    include_audio = getattr(args, 'include_audio', False)

    # Use InterHuman path when batch has InterHuman data (IH/Rel tokens); else HumanML3D (<Motion_i>)
    use_ih = any(batch_interhuman[i] is not None for i in range(len(batch_interhuman)))
    motion_repr = 'interhuman' if use_ih else getattr(args, 'motion_repr_type', 'humanml3d')
    if use_ih:
        args.motion_repr_type = 'interhuman'   # ensure we use InterHuman prompts
    print(f"motion_repr_type: {motion_repr} (batch has InterHuman: {use_ih})")

    input_ids, target_ids, attn_mask = process_batch_Salsa(
        tokenizer=tokenizer,
        batch_aux_info=levels,
        batch_ms_desc_L=ms_desc_L,
        batch_ms_des_F=ms_des_F,
        batch_vq_tokens_L=vq_tokens_L,
        batch_vq_tokens_F=vq_tokens_F,
        batch_audio_tokens=audio_tokens,
        max_tgt_len=700,
        current_batch_task=current_task,
        motion_repr_type=motion_repr,
        batch_interhuman_data=batch_interhuman,
        include_audio=include_audio,
    )

    # Detokenize (token ids → text) so we can verify what goes into the model in readable form.
    # Split at prompt/target boundary: target_ids use -100 for prompt, real token ids for target.
    t0 = target_ids[0]
    # Find first non-(-100) index = where target starts (not last -100 which could be padding)
    target_start_idx = (t0 != -100).nonzero(as_tuple=True)[0]
    prompt_len = target_start_idx[0].item() if len(target_start_idx) > 0 else len(t0)
    
    # Debug: check target_ids content
    n_prompt = (t0 == -100).sum().item()
    n_target = (t0 != -100).sum().item()
    print(f"\n[Debug] target_ids[0]: {n_prompt} prompt tokens (-100), {n_target} target tokens (real ids), prompt_len={prompt_len}")
    
    input_token_ids = input_ids[0][:prompt_len]
    target_token_ids = input_ids[0][prompt_len:]
    input_text = tokenizer.decode(input_token_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_token_ids, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("DETOKENIZED FOR VERIFICATION (token ids → readable text)")
    print("This is what goes into the model for training, in human-readable form.")
    print("=" * 60)
    print("\n--- Input (detokenized) ---")
    print(f"  [token count: {len(input_token_ids)}]")
    print(input_text[:2500] + ("... [truncated]" if len(input_text) > 2500 else ""))
    print("\n--- Target (detokenized) ---")
    print(f"  [token count: {len(target_token_ids)}]")
    print(target_text[:2500] + ("... [truncated]" if len(target_text) > 2500 else ""))

    # Full sequence: what the model actually receives (prompt + target concatenated)
    full_sequence_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print("\n--- Full sequence (what model receives: prompt + target concatenated) ---")
    print(f"  [total token count: {len(input_ids[0])}]")
    print(full_sequence_text[:3000] + ("... [truncated]" if len(full_sequence_text) > 3000 else ""))

    print("\n--- Shapes ---")
    print(f"input_ids {input_ids.shape}, target_ids {target_ids.shape}, attention_mask {attn_mask.shape}")
    print("main2 done.")


if __name__ == '__main__':
    main()
    # main2()
