import torch

from tiger_rec.semantic_id.rqvae import (
    MultiViewRQVAE,
    inverse_frequency_weights,
    reset_dead_codebooks,
    reset_dead_codebooks_error_aware,
)


def test_rqvae_forward():
    model = MultiViewRQVAE(
        text_dim=8,
        collab_dim=6,
        hidden_dim=16,
        latent_dim=12,
        num_levels=3,
        codebook_size=8,
        gate_hidden_dim=16,
    )
    text = torch.randn(5, 8)
    collab = torch.randn(5, 6)
    output = model(text, collab, sample_weights=torch.ones(5), topk=3)
    assert output["quantized"].codes.shape == (5, 3)
    assert output["quantized"].topk_codes.shape == (5, 3, 3)
    assert torch.isfinite(output["reconstruction_loss"])
    assert torch.isfinite(output["quantized"].hard_usage_loss)
    assert torch.isfinite(output["quantized"].joint_collision_loss)
    assert model.quantizer.usage_ema.sum().item() > 0


def test_inverse_frequency_weights_emphasize_tail():
    weights = inverse_frequency_weights([100, 1], power=0.5, clip=20)
    assert weights[1] > weights[0]


def test_reset_dead_codebooks_reinitializes_unused_codes():
    model = MultiViewRQVAE(
        text_dim=8,
        collab_dim=6,
        hidden_dim=16,
        latent_dim=12,
        num_levels=2,
        codebook_size=4,
        gate_hidden_dim=16,
    )
    latent = torch.randn(6, 12)
    usage = [torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)]
    before = [codebook.detach().clone() for codebook in model.quantizer.codebooks]
    reset = reset_dead_codebooks(model, latent, usage, noise=0.0)
    assert reset == 8
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, model.quantizer.codebooks))


def test_error_aware_reset_replaces_dead_codes_with_high_error_samples():
    torch.manual_seed(7)
    model = MultiViewRQVAE(
        text_dim=8,
        collab_dim=6,
        hidden_dim=16,
        latent_dim=12,
        num_levels=2,
        codebook_size=8,
        gate_hidden_dim=16,
    )
    latent = torch.randn(5, 12)
    before = [codebook.detach().clone() for codebook in model.quantizer.codebooks]
    counts = reset_dead_codebooks_error_aware(model, latent, threshold=0.0, noise=0.0)
    assert counts["total"] > 0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, model.quantizer.codebooks))


def test_collaborative_only_fusion_forward():
    model = MultiViewRQVAE(
        text_dim=8,
        collab_dim=6,
        hidden_dim=16,
        latent_dim=12,
        num_levels=2,
        codebook_size=8,
        fusion_type='collab_only',
        gate_hidden_dim=16,
    )
    output = model(
        torch.randn(5, 8), torch.randn(5, 6),
        sample_weights=torch.ones(5), topk=2,
    )
    assert output['reconstruction'].shape == (5, 16)
    assert torch.isfinite(output['reconstruction_loss'])
