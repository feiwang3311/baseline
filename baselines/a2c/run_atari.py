#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn, test
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, DnnPolicy

def train(env_id, num_frames, seed, policy, lrschedule, num_cpu, save_dir=None, test_path=None, model_dir=None):
    num_timesteps = int(num_frames * 1.1) # Change for SAT: no need to divide by 4
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and 
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank))) #, allow_early_resets=True
            gym.logger.setLevel(logging.WARN)
            return env # Change for SAT: wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'dnn':
        policy_fn = DnnPolicy
    if test_path is not None:
        test(policy_fn, env, save_dir, model_dir, test_path, env_id)
    else:
        learn(policy_fn, env, seed, total_timesteps=num_timesteps, lrschedule=lrschedule)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='gym_sat_Env-v0') # Change for SAT: use gym_sat_Env-v0 was BreakoutNoFrameskip-v4
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnn'], default='cnn') # Change for SAT: use dnn as default
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=1) # Change for SAT: use 1, was 40
    # Comments by Fei: arguments for testing mode
    parser.add_argument('--save-dir', help='where is the model saved', default=None)
    parser.add_argument('--test_path', help='where is test files saved', default=None)
    parser.add_argument('--model-dir', help='which model to test on', default=None)
    args = parser.parse_args()
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, 
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=16, 
        save_dir = args.save_dir, test_path = args.test_path, model_dir = args.model_dir) 

if __name__ == '__main__':
    main()
