# utils/custom_optimizer.py
import math
import torch
from torch.optim.optimizer import Optimizer


class CustomOptimizer(Optimizer):
    """
    自定义优化器实现 - 结合了SGD和Adam的特点

    参数:
        params (iterable): 待优化的参数迭代器
        lr (float, optional): 学习率 (默认: 0.01)
        beta1 (float, optional): 一阶矩估计的指数衰减率 (默认: 0.9)
        beta2 (float, optional): 二阶矩估计的指数衰减率 (默认: 0.999)
        epsilon (float, optional): 数值稳定性的小常数 (默认: 1e-8)
        weight_decay (float, optional): L2权重衰减 (默认: 0)
        momentum (float, optional): 动量因子 (默认: 0)
        nesterov (bool, optional): 是否使用Nesterov动量 (默认: False)
    """

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0, momentum=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon,
                        weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        super(CustomOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化

        参数:
            closure (callable, optional): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 处理稀疏梯度
                if grad.is_sparse:
                    raise RuntimeError(
                        'CustomOptimizer does not support sparse gradients')

                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩估计 (动量)
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    # 二阶矩估计 (梯度平方的移动平均)
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    # 额外的动量缓存
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 应用权重衰减
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # 更新一阶矩和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差校正后的二阶矩估计
                denom = (exp_avg_sq.sqrt() /
                         math.sqrt(bias_correction2)).add_(epsilon)

                # 计算学习率
                step_size = lr / bias_correction1

                # 应用动量
                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(
                        exp_avg, denom, value=step_size)

                    if nesterov:
                        p.add_(buf, alpha=momentum).addcdiv_(
                            exp_avg, denom, value=-step_size)
                    else:
                        p.add_(buf, alpha=-1)
                else:
                    # 标准更新
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
