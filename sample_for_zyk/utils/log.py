from torch.utils.tensorboard import SummaryWriter
import numpy as np

def record_res(writer: SummaryWriter, res_ls, step):
    point_p_ls, point_f_ls, point_r_ls = [], [], []
    range_p_ls, range_f_ls, range_r_ls = [], [], []

    for i, (point_res, range_res) in enumerate(res_ls):
        # writer.add_scalars(f"Entity Result/{i}/Point Res", point_res, step)
        # writer.add_scalars(f"Entity Result/{i}/Range Res", range_res, step)

        point_p_ls.append(point_res['p'])
        point_r_ls.append(point_res['r'])
        point_f_ls.append(point_res['f'])
        range_p_ls.append(range_res['p'])
        range_r_ls.append(range_res['r'])
        range_f_ls.append(range_res['f'])

    point_avg_res = {
        'p': np.average(point_p_ls), 
        'r': np.average(point_r_ls), 
        'f': np.average(point_f_ls), 
    }
    range_avg_res = {
        'p': np.average(range_p_ls), 
        'r': np.average(range_r_ls), 
        'f': np.average(range_f_ls), 
    }
    
    writer.add_scalars(f"Avg Result/Point Res", point_avg_res, step)
    writer.add_scalars(f"Avg Result/Range Res", range_avg_res, step)
    
    return point_avg_res, range_avg_res