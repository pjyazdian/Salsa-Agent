import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--is-baseline', action='store_true', help='whether to use baseline model architecture')
    parser.add_argument('--is-MDM', action='store_true', help='use MDM cache (no MotionScript); default when not passed')
    parser.add_argument('--no-MDM', action='store_true', help='use non-MDM cache (with MotionScript); overrides --is-MDM')
    parser.add_argument('--motion-repr-type', type=str, default='humanml3d', choices=['humanml3d', 'interhuman'],
                        help='humanml3d: HumanML3D + <Motion_i>; interhuman: InterHuman/Rel + <IH_i>, <Rel_i>')
    parser.add_argument('--nb-ih-code', type=int, default=512, help='InterHuman codebook size (if motion-repr-type=interhuman)')
    parser.add_argument('--nb-rel-code', type=int, default=512, help='Relationship codebook size (if motion-repr-type=interhuman)')
    parser.add_argument('--include-motionscript', action='store_true', help='Add MotionScript tokens; set from non-MDM when training.')
    parser.add_argument('--include-audio', action='store_true', help='Add audio tokens (<Audio_0>..) to tokenizer for training with audio modality')

    ## LLM 
    parser.add_argument('--llm-backbone', type=str, default='google/gemma-2-2b-it',
                        help='HF CausalLM id, or gpt_ablation for non-LLM Option-A control (random-init GPT2)')
    parser.add_argument('--lora-r-t2m', type=int, default=64, help='lora_r for t2m')
    parser.add_argument('--lora-alpha-t2m', type=int, default=64, help='lora_alpha for t2m')
    parser.add_argument('--lora-r-m2t', type=int, default=32, help='lora_r for m2t')
    parser.add_argument('--lora-alpha-m2t', type=int, default=32, help='lora_alpha for m2t')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='lora_dropout')

    ## Non-LLM GPT ablation (used only when --llm-backbone gpt_ablation; defaults ≈ P_train_S2)
    parser.add_argument('--gpt-n-layer', type=int, default=12, help='gpt_ablation: number of transformer layers')
    parser.add_argument('--gpt-n-embd', type=int, default=768, help='gpt_ablation: hidden size')
    parser.add_argument('--gpt-n-head', type=int, default=12, help='gpt_ablation: attention heads')
    parser.add_argument('--gpt-n-positions', type=int, default=1024, help='gpt_ablation: max sequence length')
    parser.add_argument('--gpt-dropout', type=float, default=0.1, help='gpt_ablation: dropout')
    parser.add_argument('--nb-audio-code', type=int, default=4096, help='gpt_ablation: audio codebook size in compact vocab')

    ## dataloader  
    parser.add_argument('--parent-dir', type=str, default='.', help='parent directory for data and checkpoints')
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='experiments', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    
    ## MotionLLM training (aligned with Motion-Agent where applicable)
    parser.add_argument('--learning-rate', type=float, dest='llm_lr', default=1e-5, help='learning rate for MotionLLM (Motion-Agent uses 1e-5); train script uses this as lr when running LLM training')
    parser.add_argument('--epochs', type=int, default=500, help='epochs for stage-1 multi-task training (Motion-Agent uses 500 for t2m)')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoint every N epochs')
    parser.add_argument('--train-batch-size', type=int, default=4, help='batch size for MotionLLM training (Motion-Agent uses 6; 4–6 is typical)')
    parser.add_argument('--use-wandb', action='store_true', help='log to wandb')
    parser.add_argument('--wandb-project', type=str, default='Salsa-LLM', help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run name (default: pretrain_all for task none/all, else <task>_v3)')
    parser.add_argument('--training-task', '--task', type=str, dest='task', default='none', help='task: none = all tasks (stage 1); or caption_to_motion, leader_to_follower, etc. for stage-2/single-task')
    parser.add_argument('--resume-ckpt', type=str, default=None, help='path to checkpoint to resume (e.g. stage-1 best or pretrained MotionLLM)')
    parser.add_argument('--save-dir', type=str, default=None, help='directory to save checkpoints (default: output_trained/<wandb_run_name>)')
    parser.add_argument('--lmdb-dir', type=str, default='dataset_processed_New/lmdb_Salsa_pair/lmdb_train', help='LMDB train directory (must match cache created by demo.py --create_cache_only)')

    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')

    args, unknown = parser.parse_known_args()
    print(f"Unkown arguments detected:\n{unknown}")
    return args