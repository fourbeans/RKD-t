import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_entropy_weights_temps(student_logits, T_min=2.0, T_max=4.0, eps=1e-8):
    """
    根据每个时间步学生logits计算：
    - 平均熵
    - 归一化权重（熵越小→权重大）
    - 对应的温度（熵越小→温度低）

    Args:
        student_logits: [T, N, C]
    Returns:
        weights: [T] 时间步权重（归一化）
        temps: [T] 每个时间步的蒸馏温度
        entropy_t: [T] 每个时间步的平均熵（可用于调试）
    """
    T, N, C = student_logits.shape
    probs = F.softmax(student_logits, dim=-1)  # [T,N,C]
    entropy_t = -(probs * torch.log(probs + eps)).sum(dim=-1).mean(dim=1)  # [T]

    # min-max normalize
    ent_min, ent_max = entropy_t.min(), entropy_t.max()
    norm_H = (entropy_t - ent_min) / (ent_max - ent_min + eps)  # [T], 0=低熵,1=高熵

    # 权重：1 - norm_H
    weights = 1.0 - norm_H
    weights = weights / (weights.sum() + eps)  # 归一化

    # 温度：熵越大温度越高
    temps = T_min + (T_max - T_min) * norm_H  # [T]

    return weights, temps, entropy_t


def logits_loss_T(student_logits, teacher_logits, T_min=2.0, T_max=5.0):
    """
    时间步自适应 温度+权重 的 logits 蒸馏损失
    student_logits: [T, N, C]
    teacher_logits: [N, C]
    """
    T_steps, N, C = student_logits.shape
    device = student_logits.device
    teacher_logits = teacher_logits.to(device)

    weights, temps, entropy_t = compute_entropy_weights_temps(
        student_logits, T_min=T_min, T_max=T_max
    )
    # print(' weights, temps, entropy_t', weights, temps, entropy_t)

    total_loss = 0.0
    for t in range(T_steps):
        temp_t = temps[t]
        # teacher soft targets
        teacher_probs = F.softmax(teacher_logits / temp_t, dim=1)
        student_log_probs = F.log_softmax(student_logits[t] / temp_t, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp_t ** 2)
        total_loss += weights[t] * kl

    return total_loss

import torch
import torch.nn.functional as F

def log_kd_tensorboard(writer, stu_entropy, temperature, global_step, prefix="kd_monitor"):
    """
    使用已计算好的熵和温度进行 TensorBoard 记录
    Args:
        writer: SummaryWriter
        stu_entropy: [T, B]
        temperature: [T, B]
        global_step: int
        prefix: str
    """
    with torch.no_grad():
        T = stu_entropy.shape[0]

        # # 逐时间步记录
        # for t in range(T):
        #     writer.add_scalar(f'{prefix}/entropy_T{t+1}_mean', stu_entropy[t].mean().item(), global_step)
        #     writer.add_scalar(f'{prefix}/entropy_T{t+1}_std', stu_entropy[t].std().item(), global_step)
        #     writer.add_scalar(f'{prefix}/temperature_T{t+1}_mean', temperature[t].mean().item(), global_step)
        #     writer.add_scalar(f'{prefix}/temperature_T{t+1}_min', temperature[t].min().item(), global_step)
        #     writer.add_scalar(f'{prefix}/temperature_T{t+1}_max', temperature[t].max().item(), global_step)

        # 整体平均（跨时间）
        entropy_mean_overT = stu_entropy.mean(0)  # [B]
        temp_mean_overT = temperature.mean(0)     # [B]

        writer.add_scalar(f'{prefix}/entropy_overall_mean', entropy_mean_overT.mean().item(), global_step)
        writer.add_scalar(f'{prefix}/entropy_overall_std', entropy_mean_overT.std().item(), global_step)
        writer.add_scalar(f'{prefix}/temperature_overall_mean', temp_mean_overT.mean().item(), global_step)
        writer.add_scalar(f'{prefix}/temperature_overall_min', temp_mean_overT.min().item(), global_step)
        writer.add_scalar(f'{prefix}/temperature_overall_max', temp_mean_overT.max().item(), global_step)



# def logits_loss_tmp(logits_student, logits_teacher, target, base_temperature):
def logits_loss_tmp(logits_student, logits_teacher, base_temperature,
                        writer=None, global_step=None):
    # 1. calculate the student's entropy
    # print('logits_student',logits_student.shape)# [T, b, C]
    student_probs = F.softmax(logits_student, dim=-1)  # [T, b, C]
    stu_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-8), dim=-1)  # [T, b]

    entropy_norm = stu_entropy / torch.log(torch.tensor(student_probs.shape[-1], device=logits_student.device))

    temperature = base_temperature + 0.5 * (1 - 2 * entropy_norm)
    temperature = temperature.clamp(min=base_temperature - 0.5, max=base_temperature + 0.5).detach()
    # print('temperature',temperature.shape) # [T, b]

    # 2. KD with different temperature
    t_stu = logits_student / temperature.unsqueeze(-1) #[T,b,c]/[T,b,1]
    t_tea = logits_teacher / temperature.unsqueeze(-1)

    pred_student = F.softmax(t_stu, dim=-1)
    pred_teacher = F.softmax(t_tea, dim=-1)

    log_pred_student = torch.log(pred_student + 1e-8)
    loss_tmp = F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(-1).mean()#teach:[b, class_num]->[T, b, class_num]
    loss_tmp  *= (base_temperature ** 2)

    # ---- TensorBoard record
    if writer is not None and global_step is not None:
        log_kd_tensorboard(writer, stu_entropy, temperature, global_step, prefix="KD")

    return loss_tmp




def feature_loss(feature_stu, feature_tea, fun = 'mse', T = 20,Norm_fun=None):
    loss_all = 0
    for i in range(len(feature_stu)):
        if fun == 'mse':
            loss_all += F.mse_loss(feature_stu[i], feature_tea[i].detach())
        elif fun == 'l1':
            loss_all += F.l1_loss(feature_stu[i],feature_tea[i].detach())
        elif fun == 'kl':
            loss_all += KL_loss(feature_stu[i], feature_tea[i].detach(),T)
        elif fun == 'norm':
            loss_all = Norm_fun(feature_stu,feature_tea)
            return loss_all
        elif fun == "mag":
            loss_all += F.l1_loss(torch.norm(feature_stu[i]),torch.norm(feature_tea[i]))
    return loss_all

def KL_loss(feature_stu, feature_tea,T = 1 ):
    B = feature_stu.shape[0]
    feature_stu = feature_stu.reshape(B, -1)
    feature_tea = feature_tea.reshape(B, -1)
    loss = F.kl_div(F.log_softmax(feature_stu / T, dim=1), F.softmax(feature_tea / T, dim=1),reduction='batchmean') * T * T
    return loss