import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from pathlib import Path

from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum

from ..common.utils import load_config, save_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="reinvent")
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.method = args.method.lower()

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == 'screening':
        from .main.screening.run import Exhaustive_Optimizer as Optimizer 
    elif args.method == 'molpal':
        from .main.molpal.run import MolPAL_Optimizer as Optimizer
    elif args.method == 'graph_ga':
        from .main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'smiles_ga':
        from .main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif args.method == "selfies_ga":
        from .main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif args.method == "synnet":
        from .main.synnet.run import SynNet_Optimizer as Optimizer
    elif args.method == 'graph_mcts':
        from .main.graph_mcts.run import Graph_MCTS_Optimizer as Optimizer
    elif args.method == 'smiles_ahc':
        from .main.smiles_ahc.run import AHC_Optimizer as Optimizer
    elif args.method == 'smiles_aug_mem':
        from .main.smiles_aug_mem.run import AugmentedMemory_Optimizer as Optimizer 
    elif args.method == 'smiles_bar':
        from .main.smiles_bar.run import BAR_Optimizer as Optimizer 
    elif args.method == "smiles_lstm_hc":
        from .main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'selfies_lstm_hc':
        from .main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'dog_gen':
        from .main.dog_gen.run import DoG_Gen_Optimizer as Optimizer
    elif args.method == 'gpbo':
        from .main.gpbo.run import GPBO_Optimizer as Optimizer
    elif args.method == 'stoned': 
        from .main.stoned.run import Stoned_Optimizer as Optimizer
    elif args.method == "selfies_vae":
        from .main.selfies_vae.run import SELFIES_VAEBO_Optimizer as Optimizer
    elif args.method == "smiles_vae":
        from .main.smiles_vae.run import SMILES_VAEBO_Optimizer as Optimizer
    elif args.method == 'jt_vae':
        from .main.jt_vae.run import JTVAE_BO_Optimizer as Optimizer
    elif args.method == 'dog_ae':
        from .main.dog_ae.run import DoG_AE_Optimizer as Optimizer
    elif args.method == 'pasithea':
        from .main.pasithea.run import Pasithea_Optimizer as Optimizer
    elif args.method == 'dst':
        from .main.dst.run import DST_Optimizer as Optimizer        
    elif args.method == 'molgan':
        from .main.molgan.run import MolGAN_Optimizer as Optimizer
    elif args.method == 'mars':
        from .main.mars.run import MARS_Optimizer as Optimizer
    elif args.method == 'mimosa':
        from .main.mimosa.run import MIMOSA_Optimizer as Optimizer
    elif args.method == 'gflownet': # Dependency issues
        from .main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif args.method == 'gflownet_al': # Dependency issues
        from .main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif args.method == 'moldqn':
        from .main.moldqn.run import MolDQN_Optimizer as Optimizer
    elif args.method == 'reinvent':
        from .main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_transformer':
        from .main.reinvent_transformer.run_transformer import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_selfies':
        from .main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif args.method == 'graphinvent':
        from .main.graphinvent.run import GraphInvent_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")
        
    # Load optimizer
    optimizer = Optimizer(args=args)
    
    # Load algo configs
    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
    
    # Load MolScore config
    cfg = load_config(args.molscore_config)
    # Single mode
    if cfg.molscore_mode == "single":
        task = MolScore(
            model_name=cfg.model_name,
            task_config=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            add_run_dir=True,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(task.save_dir) / "args.yaml")
        save_config(cfg, Path(task.save_dir) / "molscore_args.yaml")
        with task as scorer:
            optimizer.optimize(
                oracle=scorer, 
                config=config_default, 
                seed=args.seed
            )
    # Benchmark mode
    if cfg.molscore_mode == "benchmark":
        MSB = MolScoreBenchmark(
            model_name=cfg.model_name,
            benchmark=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            add_benchmark_dir=True,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(MSB.output_dir) / "args.yaml")
        save_config(cfg, Path(MSB.output_dir) / "molscore_args.yaml")
        with MSB as benchmark:
            for task in benchmark:
                with task as scorer:
                    optimizer.optimize(
                        oracle=scorer, 
                        config=config_default, 
                        seed=args.seed
                    )
    # Curriculum mode
    if cfg.molscore_mode == "curriculum":
        task = MolScoreCurriculum(
            model_name=cfg.model_name,
            benchmark=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(task.save_dir) / "args.yaml")
        save_config(cfg, Path(task.save_dir) / "molscore_args.yaml")
        with task as scorer:
            optimizer.optimize(
                oracle=scorer, 
                config=config_default, 
                seed=args.seed
            )


if __name__ == "__main__":
    main()

