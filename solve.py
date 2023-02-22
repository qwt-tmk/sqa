import os
import json
from datetime import datetime, timezone, timedelta

from utils.graph import generate_weighted_graph
from problems.tsp import TSP

import numpy as np

def main(problem_params, solver_params):

    _problem_params = {'n':10, 'p':0.3, 'term_weights': (1,1,1,2)}
    _problem_params.update(**problem_params)

    tsp = TSP(*generate_weighted_graph(_problem_params['n'], p=_problem_params['p']))
    tsp.set_whole_qubo(*_problem_params['term_weights'])

    tau = 10000
    gamma_schedule = [ (1, gamma) for gamma in np.linspace(3.0, 1e-8, tau) ] # (MCS, gamma) のリストで横磁場のスケジュールを決める
    #save_points = np.geomspace(1, tau, 100) # このMCSを超えた時点の状態を記録できる
    save_points = list(range(0, tau+1)) #全部
    dt = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M")
    save_dir = f'./tmp/{dt}' # 保存するディレクトリ
    trotter = 40
    T = 1/trotter # T*trotter が有効模型での温度に対応する
    preanneal = True # SQAの初期状態を古典系に対するSAの結果をトロッター方向に並べたものにする. Falseの場合は全ての層がバラバラで始まる

    _solver_params = {
        'trotter' : trotter,
        "T" : T,
        "gamma_schedule" : gamma_schedule,
        "save_points" : save_points,
        "save_dir" : save_dir,
        "preanneal" : preanneal
    }

    _solver_params.update(**solver_params)
    save_dir = _solver_params['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tsp.solve_mysqa(**_solver_params)
    tsp.save(save_dir)
    tsp.plot_net(os.path.join(save_dir, 'network.png'))
    tsp.plot_qubo(os.path.join(save_dir, 'qubo.png'))
    with open(os.path.join(save_dir, 'info.json'), 'w') as fp:
        json.dump({'problem_params': _problem_params,
                      'solver_params': _solver_params},
                  fp, ensure_ascii=False,indent=4)

if __name__=='__main__':
    main({}, {})

