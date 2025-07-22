def step_decay_lr(initial_lr: float, step: int, decay_steps: int, decay_factor: float = 0.1) -> float:
    return initial_lr * (decay_factor ** (step // decay_steps))