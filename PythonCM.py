from queue import Queue
import threading

class PyCarMakerEnv:
  def __init__(self, host='127.0.0.1', port=10001, matlab_path='C:/CM_Projects/han_230706/src_cm4sl', simul_path='test_IPG_env13'):
    #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
    env_action_num = 1
    sim_action_num = env_action_num + 1

    # Env의 observation 개수와 simulink observation 개수
    env_obs_num = 13
    sim_obs_num = 13

    # 카메이커 연동 쓰레드와의 데이터 통신을 위한 큐
    self.status_queue = Queue()
    self.action_queue = Queue()
    self.state_queue = Queue()

    # 각 Env마다 1개의 카메이커 연동 쓰레드를 사용
    self.cm_thread = threading.Thread(target=cm_thread, daemon=False, args=(host,port,self.action_queue, self.state_queue, sim_action_num, sim_obs_num, self.status_queue, matlab_path, simul_path))
    self.cm_thread.start()

    self.test_num = 0

    self.traj_data_before = pd.read_csv("datasets_traj.csv")
    self.traj_data = self.traj_data_before.loc[:, ["traj_tx", "traj_ty"]].values

def cm_thread(host, port, action_queue, state_queue, action_num, state_num, status_queue, matlab_path, simul_path):
    cm_env = CMcontrolNode(host=host, port=port, action_queue=action_queue, state_queue=state_queue, action_num=action_num, state_num=state_num, matlab_path=matlab_path, simul_path=simul_path)

    while True:
        # 강화학습에서 카메이커 시뮬레이션 상태를 결정
        status = status_queue.get()
        if status == "start":
            # 시뮬레이션 시작
            # TCP/IP 로드 -> 카메이커 시뮬레이션 시작 -> 강화학습과 데이터를 주고 받는 loop
            cm_env.start_sim()
        elif status == "stop":
            # 시뮬레이션 종료
            cm_env.stop_sim()
        elif status == "finish":
            # 프로세스 종료
            cm_env.kill()
            break
        else:
            time.sleep(1)