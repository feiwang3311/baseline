#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
#from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy

def train(env_id, num_frames, seed, num_cpu, save_dir, test_path, model_dir, save_interval):
    num_timesteps = int(num_frames * 1.1) # Change for SAT: no need to divide by 4
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env # Change for SAT: wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu, save_dir = save_dir, save_interval = save_interval)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='gym_sat_Env-v0') # Change for SAT: use gym_sat_Env-v0 was BreakoutNoFrameskip-v4
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=1) # Change for SAT: use 1, was 40
    # Comments by Fei: arguments for testing mode
    parser.add_argument('--save-dir', help='where is the model saved', default=None)
    parser.add_argument('--test_path', help='where is test files saved', default=None)
    parser.add_argument('--model-dir', help='which model to test on', default=None)
    parser.add_argument('--save_interval', help='how often to save models', default=None)
    args = parser.parse_args()    
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, num_cpu=32, 
        save_dir = args.save_dir, test_path = args.test_path, model_dir = args.model_dir, save_interval = args.save_interval)

if __name__ == '__main__':
    main()
