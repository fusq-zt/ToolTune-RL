"""Analytic checks through the installed trainer's actual loss implementation."""

import torch


def check_loss_contract(trainer):
    method = trainer._get_per_token_logps_and_entropies
    beta, correction = trainer.beta, trainer.vllm_importance_sampling_correction
    previous_accum = getattr(trainer, "current_gradient_accumulation_steps", None)
    trainer.beta = 0.0
    trainer.vllm_importance_sampling_correction = False
    trainer.current_gradient_accumulation_steps = 1
    device = trainer.accelerator.device

    def inputs(batch, mask, advantages):
        return dict(
            prompt_ids=torch.ones((batch, 1), device=device, dtype=torch.long),
            prompt_mask=torch.ones((batch, 1), device=device, dtype=torch.long),
            completion_ids=torch.ones((batch, 2), device=device, dtype=torch.long),
            completion_mask=torch.ones((batch, 2), device=device, dtype=torch.long),
            tool_mask=torch.tensor(mask, device=device),
            advantages=torch.tensor(advantages, device=device),
            old_per_token_logps=torch.zeros((batch, 2), device=device),
        )

    try:
        logps = torch.zeros((1, 2), device=device, requires_grad=True)
        trainer._get_per_token_logps_and_entropies = lambda *a, **kw: (
            logps,
            torch.ones_like(logps),
        )
        loss = trainer._compute_loss(trainer.model, inputs(1, [[1, 0]], [1.0]))
        loss.backward()
        assert (
            logps.grad[0, 0] != 0 and logps.grad[0, 1] == 0
        ), "Environment token entered policy loss"
        shared = torch.tensor(0.0, device=device, requires_grad=True)
        suffix = torch.zeros(2, device=device, requires_grad=True)
        lp = torch.stack([torch.stack([shared, suffix[i]]) for i in range(2)])
        trainer._get_per_token_logps_and_entropies = lambda *a, **kw: (
            lp,
            torch.ones_like(lp),
        )
        loss = trainer._compute_loss(
            trainer.model, inputs(2, [[1, 1], [1, 1]], [1.0, -0.5])
        )
        loss.backward()
        expected = -0.5 / (2 * trainer.max_completion_length)
        assert torch.isclose(
            shared.grad, torch.tensor(expected, device=device)
        ), "Shared-prefix gradient miscounted"
        return {
            "environment_direct_gradient": 0.0,
            "policy_gradient_nonzero": True,
            "shared_gradient": float(shared.grad),
            "expected_shared_gradient": expected,
            "trainer_loss_type": trainer.loss_type,
        }
    finally:
        trainer._get_per_token_logps_and_entropies = method
        trainer.beta = beta
        trainer.vllm_importance_sampling_correction = correction
        if previous_accum is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = previous_accum
        trainer._metrics["train"].clear()
        trainer._metrics["eval"].clear()
