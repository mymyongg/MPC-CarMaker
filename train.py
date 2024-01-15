"""
학습 코드 예제
1. 카메이커 연동 환경을 불러온다
    1-1. 여러 대의 카메이커를 실행하기 위해 SubprocVecEnv를 이용하여 멀티프로세싱이 가능한 환경 로드
2. 학습에 사용할 RL 모델(e.g. PPO)을 불러온다.
3. 학습을 진행한다. x
    3-1. total_timesteps 수를 변화시켜 충분히 학습하도록 한다.
4. 학습이 완료된 후 웨이트 파일(e.g. model.pkl)을 저장한다.
"""
from stable_baselines3 import SAC
from callbacks import getBestRewardCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
import os
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from collections import namedtuple
from callbacks import getBestRewardCallback
from RL.rl_carmaker_env import RLCarMakerEnv

# GPU를 사용할 수 있는지 확인합니다.
if torch.cuda.is_available():
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda:" + str(device_id))
    print(f"Using GPU device ID {device_id}.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")
# device = torch.device("cpu")
# print("No GPU available, using CPU.")


def make_env(rank, seed=0):

    def _init():
        matlab_path = "/home/seoyeon/mpc_ws/RWS/JX1_102/src_cm4sl"
        simul_path = "pythonCtrl_mpc"
        save_data = False
        env = RLCarMakerEnv(port=10000 + rank, matlab_path=matlab_path, simul_path=simul_path, save_data=save_data)  # 모니터 같은거 씌워줘야 할거임
        # env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init


def main():
    num_proc = 1
    naming = "slalom_1213"
    log_dir = "models/"
    args = {}
    args['prefix'] = naming
    args['alg'] = 'sac'
    Args = namedtuple('Args', ['prefix', 'alg'])
    args = Args(prefix=naming, alg='sac')


    # Set environment
    env = SubprocVecEnv([make_env(i) for i in range(num_proc)])
    env = VecMonitor(env, log_dir+args.prefix)
    bestRewardCallback = getBestRewardCallback(args)
    
    # Set model and run learning
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join("tensorboard/{}".format(naming)))

    try:
        model.learn(total_timesteps=10000*50, log_interval=10, callback=bestRewardCallback)

    except KeyboardInterrupt:
        print("Learning interrupted. Will save the model now.")

    finally:
        print("Saving model..")
        model.save("models//model_{}_last.pkl".format(naming))
        print("Model saved.")


if __name__ == '__main__':
    main()